"""
generate_all_charts.py -- charts and numbers of Chapter 4 (TSA): seasonality and forecasting (SARIMA, TBATS, Prophet)
=====================================================================================================================
Course data (tsa_data.py), chart style (tsa_style.py), statsmodels (SARIMAX, MSTL, ETS), the tbats and prophet
packages (optional: the functions that need them check that they are installed). Every number on the slides
comes from here.
  * seasonal data   -- Romanian GDP (not adjusted), retail trade, industrial production, tourism nights, HICP
                       inflation (Eurostat) and daily electricity load (ENTSO-E); seasonal plots;
  * seasonal roots  -- the roots of 1 - L^s on the unit circle; ACF of log GDP after regular and seasonal
                       differencing; the HEGY test (critical values by simulation), OCSB and Canova-Hansen decisions;
  * SARIMA theory   -- theoretical ACF and PACF of seasonal AR, seasonal MA and the airline model;
  * airline model   -- Box and Jenkins' airline passengers: identification, estimation, forecasts;
  * GDP case        -- SARIMA models of Romanian log GDP (not adjusted): AICc and BIC table, diagnostics, forecasts;
                       Eurostat seasonally adjusted (SCA) and unadjusted (NSA) series;
  * calendar        -- Fourier approximation of a seasonal pattern; Orthodox Easter and working-day effects in
                       Romanian retail trade (regression with SARIMA errors);
  * multiple seas.  -- hourly and daily electricity load in Romania: ACF, MSTL, holidays;
  * models          -- dynamic harmonic regression, TBATS, Prophet, SARIMA, ETS and the seasonal naive method on daily
                       load; time-series cross-validation (rolling origin), MAE, RMSE, MASE, Diebold-Mariano tests
                       with the Harvey-Leybourne-Newbold correction, forecast combination;
  * GDP evaluation  -- rolling-origin comparison of SARIMA, ETS and the seasonal naive method for quarterly GDP.
Output: charts/tsa_ch4_*.pdf/.png, Quantlets/Ch_04/ch4_numbers.json, ch4_load_cv.csv, ch4_gdp_models.csv,
        ch4_ro_load_hourly.csv (hourly load of Romania, ENTSO-E, the extract used in the chapter)
References: Huang and Petukhina (2022), Ch. 4-5; Hyndman and Athanasopoulos, Forecasting: Principles and Practice
(3rd ed.), Ch. 5, 9, 10, 12, 13; Box, Jenkins and Reinsel (2008), Ch. 9; Hylleberg et al. (1990); De Livera, Hyndman
and Snyder (2011); Taylor and Letham (2018); Diebold and Mariano (1995); Harvey, Leybourne and Newbold (1997).
Run:  python3 Quantlets/Ch_04/generate_all_charts.py          (about 10 minutes: TBATS is refitted at 26 origins)
Time Series Analysis - Daniel Traian PELE
"""

import json
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
from tsa_data import read_eurostat                                       # noqa: E402
import tsa_style as st                                                   # noqa: E402
from statsmodels.tsa.stattools import acf, pacf                          # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX                   # noqa: E402
from statsmodels.tsa.arima_process import ArmaProcess                    # noqa: E402
from statsmodels.tsa.holtwinters import ExponentialSmoothing             # noqa: E402
from statsmodels.tsa.seasonal import MSTL                                # noqa: E402
from statsmodels.regression.linear_model import OLS                      # noqa: E402
from dateutil.easter import easter, EASTER_ORTHODOX                      # noqa: E402

warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)
SEED = 2026
GDP_NSA = ('namq_10_gdp', 'Q.CLV10_MEUR.NSA.B1GQ.RO')       # real GDP, chain-linked volumes (2010), not adjusted
GDP_SCA = ('namq_10_gdp', 'Q.CLV10_MEUR.SCA.B1GQ.RO')       # the same, seasonally and calendar adjusted
RETAIL = ('sts_trtu_m', 'M.VOL_SLS.G47.NSA.I21.RO')         # retail trade volume (except motor vehicles), 2021 = 100
RETAIL_SCA = ('sts_trtu_m', 'M.VOL_SLS.G47.SCA.I21.RO')
FOOD = ('sts_trtu_m', 'M.VOL_SLS.G47_FOOD.NSA.I21.RO')      # retail sale of food, beverages and tobacco
IPI = ('sts_inpr_m', 'M.PRD.B-D.NSA.I21.RO')                # industrial production (mining, manufacturing, energy)
TOUR = ('tour_occ_nim', 'M.TOTAL.NR.I551-I553.RO')          # nights spent in tourist accommodation
HICP = ('prc_hicp_minr', 'M.I15.TOTAL.RO')                  # harmonised index of consumer prices, 2015 = 100
GDP_START = '2000-01-01'
ENTSOE = 'https://www.entsoe.eu/publications/data/power-stats/{y}/monthly_hourly_load_values_{y}.csv'
LOAD_FILE = 'ch4_ro_load_hourly.csv'                        # extract: hourly load of Romania, UTC
LOAD_RAW = 'https://raw.githubusercontent.com/danpele/Time-Series-Analysis/main/Quantlets/Ch_04/' + LOAD_FILE
LOAD_YEARS = (2022, 2023, 2024, 2025, 2026)
LOAD_END = '2026-06-30'
CV_FIRST, CV_STEP, CV_H = '2025-06-30', 14, 14              # daily load: first origin, days between origins, horizon
BAND_COL = st.IDAred
MODEL_COL = {'Seasonal naive': st.Amber, 'ETS': st.Teal, 'SARIMA': st.Purple, 'DHR': st.MainBlue,
             'TBATS': st.Forest, 'Prophet': st.Orange, 'Combination': st.IDAred}


def have(pkg):
    """True if an optional package (tbats, prophet, pmdarima) is installed."""
    import importlib.util
    return importlib.util.find_spec(pkg) is not None


# =============================================================================
# DATA
# =============================================================================
def eurostat_log(key, start=None):
    """Natural log of a Eurostat series (monthly or quarterly), from `start`."""
    s = np.log(read_eurostat(*key))
    s = s.loc[start:] if start else s
    s.index.freq = None
    return s


def ro_holidays(years):
    """Romanian public holidays (Orthodox Easter and Pentecost from dateutil): date -> name."""
    out = {}
    for y in years:
        e = pd.Timestamp(easter(y, EASTER_ORTHODOX))
        day = pd.Timedelta(days=1)
        hol = {'New Year': [f'{y}-01-01', f'{y}-01-02'], 'Union Day': [f'{y}-01-24'],
               'Easter': [e - 2 * day, e, e + day], 'Labour Day': [f'{y}-05-01'], "Children's Day": [f'{y}-06-01'],
               'Pentecost': [e + 49 * day, e + 50 * day], 'Assumption': [f'{y}-08-15'], "St Andrew's Day": [f'{y}-11-30'],
               'National Day': [f'{y}-12-01'], 'Christmas': [f'{y}-12-25', f'{y}-12-26']}
        if y >= 2024:
            hol['Epiphany'] = [f'{y}-01-06', f'{y}-01-07']
        for k, v in hol.items():
            for x in v:
                out[pd.Timestamp(x)] = k
    return pd.Series(out, name='holiday').sort_index()


def load_hourly(years=LOAD_YEARS, end=LOAD_END):
    """Hourly electricity load of Romania (MW, hourly average), ENTSO-E 'monthly hourly load values'.
    Reads the extract ch4_ro_load_hourly.csv (local or from the TSA repository); otherwise downloads the yearly ENTSO-E
    files (about 40 MB each) and keeps Romania. Returns the series in UTC+2 (Romanian winter time, no clock change),
    with isolated one-hour glitches (more than 30% away from both neighbours) and missing hours interpolated."""
    s = None
    local = [os.path.join(HERE, LOAD_FILE)] + [os.path.join(d, 'Quantlets', 'Ch_04', LOAD_FILE) for d in ('.', '..', '../..', '../../..')]
    for src in [p for p in local if os.path.exists(p)] + [LOAD_RAW]:
        try:
            s = pd.read_csv(src, index_col=0, parse_dates=True).iloc[:, 0]
            break
        except Exception:
            continue
    if s is None:
        parts = []
        for y in years:
            url = ENTSOE.format(y=y)
            head = pd.read_csv(url, nrows=2, encoding='utf-8-sig', sep=None, engine='python')
            sep = '\t' if '\t' in ''.join(head.columns) or len(head.columns) == 1 else ','
            t = pd.read_csv(url, sep=sep, usecols=['DateUTC', 'CountryCode', 'Value'], encoding='utf-8-sig')
            t = t[t['CountryCode'] == 'RO']
            parts.append(pd.Series(t['Value'].astype(float).values,
                                   index=pd.to_datetime(t['DateUTC'], format='%d-%m-%Y %H:%M')))
        s = pd.concat(parts)
        s = s[~s.index.duplicated()].sort_index().rename('load_MW')
        s.index.name = 'date_utc'
        try:
            s.to_csv(os.path.join(HERE, LOAD_FILE) if os.path.isdir(HERE) else LOAD_FILE)
        except OSError:
            pass
    s = s.loc[:pd.Timestamp(end) + pd.Timedelta(hours=21)]
    s.index = s.index + pd.Timedelta(hours=2)
    s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq='h'))
    r = s / ((s.shift(1) + s.shift(-1)) / 2)
    s[(r < 0.7) | (r > 1.3)] = np.nan
    return s.interpolate().rename('load')


def load_daily(end=LOAD_END):
    """Daily mean load in GW (Romanian winter time), 1 January 2022 - 30 June 2026."""
    d = load_hourly().resample('D').mean() / 1000
    return d.loc['2022-01-01':end].rename('load')


def airline():
    """Box and Jenkins' monthly international airline passengers (thousands), 1949-1960 (R data set via statsmodels)."""
    import statsmodels.api as sm
    d = sm.datasets.get_rdataset('AirPassengers', 'datasets').data
    return pd.Series(d['value'].values.astype(float), index=pd.date_range('1949-01-01', periods=len(d), freq='MS'),
                     name='passengers')


# =============================================================================
# HELPERS
# =============================================================================
def sample_acf(x, nlags=20):
    return acf(np.asarray(x, float), nlags=nlags, fft=True)[1:]


def sample_pacf(x, nlags=20):
    return pacf(np.asarray(x, float), nlags=nlags, method='ywm')[1:]


def ljung_box(x, m=24, df=None):
    """Ljung-Box Q*(m) with df degrees of freedom (m minus the number of ARMA parameters for residuals)."""
    x = np.asarray(x, float)
    T = len(x)
    r = sample_acf(x, m)
    q = T * (T + 2) * np.sum(r ** 2 / (T - np.arange(1, m + 1)))
    df = m if df is None else df
    return {'lb': float(q), 'df': int(df), 'lb_p': float(stats.chi2.sf(q, df))}


def acf_bars(ax, r, T, color=st.MainBlue, label='_nolegend_', band=True, mark=None):
    """Bars at lags 1..len(r), the +-1.96/sqrt(T) band, and vertical marks at seasonal lags."""
    lags = np.arange(1, len(r) + 1)
    if mark:
        for m in mark:
            ax.axvline(m, color=st.Orange, lw=0.8, ls=':', label='_nolegend_')
    ax.bar(lags, r, width=0.6, color=color, label=label)
    if band:
        b = 1.96 / np.sqrt(T)
        ax.axhline(b, color=BAND_COL, ls='--', lw=0.9, label=r'$\pm 1.96/\sqrt{T}$')
        ax.axhline(-b, color=BAND_COL, ls='--', lw=0.9, label='_nolegend_')
    ax.axhline(0, color=st.DarkText, lw=0.6)
    ax.set_xlim(0.3, len(r) + 0.7)


def save(name, save_it=True):
    if save_it:
        st.check_no_grey(plt.gcf())
        st.save_fig(name)
    else:
        plt.show()


def date_axis(ax, kind='year'):
    """Readable date ticks: 'year' (one tick per year), 'month' (month initials), 'week' (day and month)."""
    if kind == 'year':
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    elif kind == 'month':
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))


def polymul(a, b):
    return np.convolve(np.asarray(a, float), np.asarray(b, float))


def sarma_process(phi=(), theta=(), Phi=(), Theta=(), s=12):
    """Multiplicative seasonal ARMA phi(L)Phi(L^s) X_t = theta(L)Theta(L^s) e_t as an ArmaProcess (expanded lags)."""
    def seas(c, sign):
        v = np.zeros(s * len(c) + 1)
        v[0] = 1
        for i, x in enumerate(c):
            v[s * (i + 1)] = sign * x
        return v
    ar = polymul(np.r_[1, -np.asarray(phi, float)], seas(Phi, -1))
    ma = polymul(np.r_[1, np.asarray(theta, float)], seas(Theta, 1))
    return ArmaProcess(ar, ma)


def fit_sarima(y, order, sorder, exog=None, trend='n'):
    """Gaussian maximum likelihood of a SARIMA (statsmodels SARIMAX, exact likelihood after differencing)."""
    return SARIMAX(np.asarray(y, float), exog=exog, order=order, seasonal_order=sorder, trend=trend).fit(
        disp=False, maxiter=200)


def aicc(res, n):
    k = len(res.params)
    return float(res.aic + 2 * k * (k + 1) / max(n - k - 1, 1))


def fourier(idx, period, K, t0='2000-01-01'):
    """Fourier terms sin(2 pi k t / period), cos(2 pi k t / period), k = 1..K, with t in days (daily data) or in
    steps (integer index)."""
    if isinstance(idx, pd.DatetimeIndex):
        t = np.asarray((idx - pd.Timestamp(t0)).days, float)
    else:
        t = np.asarray(idx, float)
    cols = {}
    for k in range(1, K + 1):
        cols[f'sin{k}_{period:g}'] = np.sin(2 * np.pi * k * t / period)
        cols[f'cos{k}_{period:g}'] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=idx if isinstance(idx, pd.DatetimeIndex) else None)


def easter_share(idx, w=10):
    """Easter regressor of a monthly series: the share of the w days before Orthodox Easter Sunday that fall in each
    month (the 'easter[w]' regressor of X-13ARIMA-SEATS, with the Orthodox date)."""
    out = pd.Series(0.0, index=idx)
    for y in sorted(set(idx.year)):
        e = pd.Timestamp(easter(y, EASTER_ORTHODOX))
        days = pd.date_range(e - pd.Timedelta(days=w), e - pd.Timedelta(days=1))
        for m in days:
            key = pd.Timestamp(m.year, m.month, 1)
            if key in out.index:
                out[key] += 1 / w
    return out


def working_days(idx):
    """Number of Monday-Friday days that are not public holidays, in each month."""
    hol = set(ro_holidays(range(idx[0].year, idx[-1].year + 1)).index)
    return pd.Series([sum(1 for d in pd.date_range(m, m + pd.offsets.MonthEnd(0)) if d.dayofweek < 5 and d not in hol)
                      for m in idx], index=idx, dtype=float)


# =============================================================================
# 1. SEASONAL DATA
# =============================================================================
def fig_series(save_it=True):
    """Six Romanian series with seasonality: GDP (NSA), retail trade, industrial production, tourism nights,
    monthly HICP inflation, daily electricity load."""
    gdp = read_eurostat(*GDP_NSA).loc[GDP_START:] / 1000
    ret = read_eurostat(*RETAIL).loc['2010':]
    ipi = read_eurostat(*IPI).loc['2010':]
    tour = read_eurostat(*TOUR).loc['2010':] / 1e6
    infl = 100 * np.log(read_eurostat(*HICP)).diff().loc['2010':]
    load = load_daily()
    panels = [('GDP, not adjusted (bn EUR, 2010 prices)', gdp, st.MainBlue), ('Retail trade volume (2021 = 100)', ret, st.IDAred),
              ('Industrial production (2021 = 100)', ipi, st.Forest), ('Nights in tourist accommodation (millions)', tour, st.Orange),
              ('HICP inflation, month on month (%)', infl, st.Purple), ('Electricity load, daily mean (GW)', load, st.Teal)]
    fig, ax = plt.subplots(3, 2, figsize=(11.0, 6.4))
    for a, (t, s, c) in zip(ax.ravel(), panels):
        a.plot(s.index, s.values, color=c, lw=0.9 if len(s) < 500 else 0.6, label=t)
        a.set_title(t, fontsize=11.5)
        if 'HICP' in t:
            a.axhline(0, color=st.DarkText, lw=0.5)
        a.xaxis.set_major_locator(mdates.YearLocator(base=4 if len(s) < 500 else 1))
        a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    save('tsa_ch4_series', save_it)
    q = gdp.groupby(gdp.index.quarter).mean()
    mt = tour.loc['2015':'2019']
    mt = mt.groupby(mt.index.month).mean()
    mi = infl.loc['2010':'2019']
    mi = mi.groupby(mi.index.month).mean()
    return {'gdp_first': str(gdp.index[0].date()), 'gdp_last': str(gdp.index[-1].date()), 'gdp_T': int(len(gdp)),
            'gdp_q': q.round(4).tolist(), 'gdp_q4_q1': float(q[4] / q[1] - 1),
            'ret_last': str(ret.index[-1].date()), 'tour_last': str(tour.index[-1].date()),
            'tour_aug_jan': float(mt[8] / mt[1]), 'tour_max_m': int(mt.idxmax()), 'tour_min_m': int(mt.idxmin()),
            'infl_month_mean': mi.round(3).tolist(), 'infl_last': str(infl.index[-1].date()),
            'load_first': str(load.index[0].date()), 'load_last': str(load.index[-1].date()), 'load_T': int(len(load)),
            'load_mean': float(load.mean())}


def fig_seasonal_plot(save_it=True):
    """Seasonal plot of tourism nights (one line per year) and the quarterly subseries of GDP."""
    tour = read_eurostat(*TOUR).loc['2012':'2025'] / 1e6
    gdp = read_eurostat(*GDP_NSA).loc['2005':] / 1000
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.9), gridspec_kw={'width_ratios': [1.25, 1]})
    years = sorted(set(tour.index.year))
    cmap = plt.get_cmap('viridis')
    for i, y in enumerate(years):
        s = tour.loc[str(y)]
        special = y in (2020, 2021)
        ax[0].plot(s.index.month, s.values, color=st.IDAred if y == 2020 else cmap(i / (len(years) - 1)),
                   lw=1.8 if special else 1.0, ls='--' if special else '-', label=str(y) if y in (2012, 2020, 2021, 2025) else '_nolegend_')
    ax[0].set_xticks(range(1, 13))
    ax[0].set_xticklabels(list('JFMAMJJASOND'))
    ax[0].set_title('Tourism nights by month, 2012-2025 (millions)')
    st.legend_outside_bottom(ax[0], ncol=4, y=-0.13)
    cols = [st.MainBlue, st.Forest, st.Orange, st.IDAred]
    for q in range(1, 5):
        s = gdp[gdp.index.quarter == q]
        x = (q - 1) + np.linspace(0.1, 0.9, len(s))
        ax[1].plot(x, s.values, color=cols[q - 1], lw=1.1, label=f'Q{q}')
        ax[1].hlines(s.mean(), q - 1 + 0.1, q - 1 + 0.9, color=st.DarkText, lw=1.6, label='quarter mean' if q == 1 else '_nolegend_')
    ax[1].set_xticks([0.5, 1.5, 2.5, 3.5])
    ax[1].set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    ax[1].set_title('GDP subseries by quarter, 2005-2026 (bn EUR)')
    st.legend_outside_bottom(ax[1], ncol=5, y=-0.13)
    plt.tight_layout()
    save('tsa_ch4_seasonal_plot', save_it)
    t19 = tour.loc['2019']
    return {'tour19_aug': float(t19.iloc[7]), 'tour19_jan': float(t19.iloc[0]), 'tour20_apr': float(tour.loc['2020-04-01']),
            'tour19_apr': float(tour.loc['2019-04-01'])}


# =============================================================================
# 2. SEASONAL DIFFERENCING AND SEASONAL UNIT ROOTS
# =============================================================================
def fig_unit_circle(save_it=True):
    """The roots of 1 - z^4 and 1 - z^12 on the unit circle, with their frequencies."""
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.6))
    th = np.linspace(0, 2 * np.pi, 400)
    for a, s, c in [(ax[0], 4, st.MainBlue), (ax[1], 12, st.IDAred)]:
        a.plot(np.cos(th), np.sin(th), color=st.DarkText, lw=0.8)
        a.axhline(0, color=st.DarkText, lw=0.4)
        a.axvline(0, color=st.DarkText, lw=0.4)
        z = np.exp(2j * np.pi * np.arange(s) / s)
        a.plot(z.real, z.imag, 'o', ms=8, color=c, label=f'roots of $1 - z^{{{s}}}$')
        a.plot([1], [0], 'o', ms=12, mfc='none', mec=st.Forest, mew=1.6, label='$z = 1$: zero frequency (trend)')
        a.plot([-1], [0], 's', ms=12, mfc='none', mec=st.Orange, mew=1.6, label=f'$z = -1$: frequency 1/2 (period 2)')
        a.set_aspect('equal')
        a.set_xlim(-1.45, 1.45)
        a.set_ylim(-1.45, 1.45)
        a.set_title(f'$s = {s}$: {s} roots of $1 - z^{{{s}}}$')
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    plt.tight_layout()
    save('tsa_ch4_unit_circle', save_it)
    return {'roots4': ['1', '-1', 'i', '-i']}


def fig_gdp_differencing(nlags=16, save_it=True):
    """Romanian log GDP (not adjusted): Delta, Delta_4 and Delta Delta_4, with their sample ACF."""
    y = eurostat_log(GDP_NSA, GDP_START)
    d1 = 100 * y.diff().dropna()
    d4 = 100 * y.diff(4).dropna()
    dd = 100 * y.diff(4).diff().dropna()
    fig, ax = plt.subplots(2, 3, figsize=(7.6, 3.6))
    out = {}
    for j, (lab, s, c) in enumerate([(r'$100\,\Delta \ln Y_t$ (quarter on quarter)', d1, st.MainBlue),
                                     (r'$100\,\Delta_4 \ln Y_t$ (year on year)', d4, st.Forest),
                                     (r'$100\,\Delta\Delta_4 \ln Y_t$', dd, st.Purple)]):
        ax[0, j].plot(s.index, s.values, color=c, lw=1.0)
        ax[0, j].axhline(0, color=st.DarkText, lw=0.5)
        ax[0, j].set_title(lab, fontsize=10)
        ax[0, j].xaxis.set_major_locator(mdates.YearLocator(8))
        ax[0, j].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        r = sample_acf(s.values, nlags)
        acf_bars(ax[1, j], r, len(s), color=c, mark=[4, 8, 12, 16])
        ax[1, j].set_ylim(-1, 1)
        ax[1, j].set_xlabel('lag (quarters)')
        out[['d1', 'd4', 'dd'][j]] = {'r': r[:8].round(4).tolist(), 'T': int(len(s)), 'sd': float(s.std())}
    ax[1, 0].plot([], [], color=st.Orange, ls=':', label='seasonal lags 4, 8, 12, 16')
    st.fig_legend_bottom(fig, ncol=2, y=-0.01)
    plt.tight_layout()
    save('tsa_ch4_gdp_diff', save_it)
    out['r4_d1'] = out['d1']['r'][3]
    out['r8_d1'] = out['d1']['r'][7]
    return out


def hegy_stats(y, k=0, trend=True):
    """HEGY regression for quarterly data (Hylleberg et al., 1990):
    D4 y_t = pi1 y1_{t-1} + pi2 y2_{t-1} + pi3 y3_{t-2} + pi4 y3_{t-1} + const + 3 seasonal dummies (+ trend)
             + sum_{j=1..k} a_j D4 y_{t-j} + e_t,
    y1 = (1 + L + L^2 + L^3) y, y2 = -(1 - L + L^2 - L^3) y, y3 = -(1 - L^2) y.
    Returns t(pi1), t(pi2), F(pi3 = pi4 = 0) and the number of observations."""
    y = np.asarray(y, float)
    n = len(y)
    L = lambda x, j: np.r_[np.full(j, np.nan), x[:n - j]]
    y1 = y + L(y, 1) + L(y, 2) + L(y, 3)
    y2 = -(y - L(y, 1) + L(y, 2) - L(y, 3))
    y3 = -(y - L(y, 2))
    d4 = y - L(y, 4)
    X = [L(y1, 1), L(y2, 1), L(y3, 2), L(y3, 1), np.ones(n)]
    q = np.arange(n) % 4
    X += [(q == j).astype(float) for j in (1, 2, 3)]
    if trend:
        X.append(np.arange(n, dtype=float))
    X += [L(d4, j) for j in range(1, k + 1)]
    X = np.column_stack(X)
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(d4)
    r = OLS(d4[ok], X[ok]).fit()
    F = float(r.f_test(np.eye(X.shape[1])[[2, 3]]).fvalue)
    return {'t1': float(r.tvalues[0]), 't2': float(r.tvalues[1]), 'F34': F, 'n': int(ok.sum()), 'aic': float(r.aic)}


def hegy_critical(n, nrep=4000, trend=True, seed=SEED):
    """5% critical values of the HEGY statistics by simulation under the null D4 y_t = e_t (seasonal random walk),
    with the same sample size and deterministic terms."""
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(nrep):
        e = rng.normal(size=n + 4)
        y = np.zeros(n + 4)
        for t in range(4, n + 4):
            y[t] = y[t - 4] + e[t]
        h = hegy_stats(y, 0, trend)
        res.append((h['t1'], h['t2'], h['F34']))
    res = np.array(res)
    return {'t1': float(np.quantile(res[:, 0], 0.05)), 't2': float(np.quantile(res[:, 1], 0.05)),
            'F34': float(np.quantile(res[:, 2], 0.95)), 'nrep': nrep}


def seasonal_tests(save_csv=True):
    """HEGY on Romanian log GDP (not adjusted, 2000-2019); OCSB and Canova-Hansen decisions (pmdarima nsdiffs) and
    the KPSS-based number of regular differences for five series, sample 2005-2019 (before the pandemic)."""
    y = eurostat_log(GDP_NSA, '2000-01-01').loc[:'2019-10-01']
    ks = {k: hegy_stats(y.values, k)['aic'] for k in range(0, 5)}
    k = int(min(ks, key=ks.get))
    h = hegy_stats(y.values, k)
    cv = hegy_critical(h['n'])
    out = {'hegy': h, 'hegy_k': k, 'hegy_cv': cv, 'hegy_first': str(y.index[0].date()), 'hegy_last': str(y.index[-1].date())}
    rows = []
    if have('pmdarima'):
        import pmdarima as pm
        for name, key, m in [('GDP', GDP_NSA, 4), ('Retail trade', RETAIL, 12), ('Industrial production', IPI, 12),
                             ('Tourism nights', TOUR, 12), ('HICP', HICP, 12)]:
            x = eurostat_log(key, '2005-01-01').loc[:'2019-12-31']
            rows.append({'series': name, 'm': m, 'ocsb': int(pm.arima.nsdiffs(x.values, m=m, test='ocsb')),
                         'ch': int(pm.arima.nsdiffs(x.values, m=m, test='ch')),
                         'd_kpss': int(pm.arima.ndiffs(x.values, test='kpss'))})
    out['table'] = rows
    if save_csv and rows:
        pd.DataFrame(rows).to_csv(os.path.join(HERE, 'ch4_seasonal_tests.csv'), index=False)
    return out


# =============================================================================
# 3. SARIMA THEORY
# =============================================================================
def fig_theory_acf(nlags=40, save_it=True):
    """Theoretical ACF and PACF of SAR(1)_12 (Phi = 0.8), SMA(1)_12 (Theta = -0.6) and the airline model
    (theta = -0.4, Theta = -0.6) for the stationary part w_t = Delta Delta_12 y_t."""
    models = [('SAR(1)$_{12}$, $\\Phi = 0.8$', dict(Phi=[0.8]), st.MainBlue),
              ('SMA(1)$_{12}$, $\\Theta = -0.6$', dict(Theta=[-0.6]), st.Forest),
              ('Airline: $\\theta = -0.4$, $\\Theta = -0.6$', dict(theta=[-0.4], Theta=[-0.6]), st.IDAred)]
    fig, ax = plt.subplots(2, 3, figsize=(11.0, 4.8))
    out = {}
    for j, (t, kw, c) in enumerate(models):
        p = sarma_process(s=12, **kw)
        r = p.acf(nlags + 1)[1:]
        pa = p.pacf(nlags + 1)[1:]
        for i, (v, lab) in enumerate([(r, 'ACF'), (pa, 'PACF')]):
            ax[i, j].bar(np.arange(1, nlags + 1), v, width=0.6, color=c)
            ax[i, j].axhline(0, color=st.DarkText, lw=0.6)
            for m in (12, 24, 36):
                ax[i, j].axvline(m, color=st.Orange, lw=0.8, ls=':')
            ax[i, j].set_ylim(-0.75, 0.9)
            ax[i, j].set_title(f'{lab}: {t}', fontsize=11)
        ax[1, j].set_xlabel('lag (months)')
        out[['sar', 'sma', 'airline'][j]] = {'acf': r[:37].round(4).tolist(), 'pacf': pa[:37].round(4).tolist()}
    ax[1, 0].plot([], [], color=st.Orange, ls=':', label='seasonal lags 12, 24, 36')
    st.fig_legend_bottom(fig, ncol=1, y=-0.01)
    plt.tight_layout()
    save('tsa_ch4_theory_acf', save_it)
    return out


# =============================================================================
# 4. THE AIRLINE MODEL
# =============================================================================
def fig_airline(nlags=36, save_it=True):
    """Airline passengers: the series, its log, and the ACF and PACF of w_t = Delta Delta_12 ln y_t."""
    y = airline()
    ly = np.log(y)
    w = ly.diff(12).diff().dropna()
    fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.5))
    ax[0].plot(y.index, y.values, color=st.MainBlue, lw=1.1, label='passengers (thousands)')
    a2 = ax[0].twinx()
    a2.plot(ly.index, ly.values, color=st.Orange, lw=1.0, label='log scale (right axis)')
    a2.spines['right'].set_visible(True)
    ax[0].set_title('Airline passengers, 1949-1960')
    acf_bars(ax[1], sample_acf(w, nlags), len(w), color=st.Purple, mark=[12, 24, 36])
    acf_bars(ax[2], sample_pacf(w, nlags), len(w), color=st.Purple, band=False, mark=[12, 24, 36])
    b = 1.96 / np.sqrt(len(w))
    ax[2].axhline(b, color=BAND_COL, ls='--', lw=0.9)
    ax[2].axhline(-b, color=BAND_COL, ls='--', lw=0.9)
    ax[1].set_title(r'ACF of $\Delta\Delta_{12}\ln y_t$')
    ax[2].set_title(r'PACF of $\Delta\Delta_{12}\ln y_t$')
    for a in ax[1:]:
        a.set_xlabel('lag (months)')
        a.set_ylim(-0.5, 0.5)
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    h3, l3 = ax[1].get_legend_handles_labels()
    st.fig_legend_bottom(fig, h1 + h2 + h3, l1 + l2 + l3, ncol=3, y=0.0)
    plt.tight_layout()
    save('tsa_ch4_airline', save_it)
    r = sample_acf(w, 24)
    return {'T': int(len(y)), 'Tw': int(len(w)), 'r1': float(r[0]), 'r12': float(r[11]), 'r11': float(r[10]),
            'r13': float(r[12]), 'r3': float(r[2]), 'band': float(1.96 / np.sqrt(len(w)))}


def fig_airline_forecast(save_it=True):
    """The airline model SARIMA(0,1,1)(0,1,1)_12 for ln y on 1949-1958, forecasts for 1959-1960 with 95% intervals,
    compared with the seasonal naive forecast; full-sample estimates."""
    y = airline()
    ly = np.log(y)
    tr, te = ly.loc[:'1958-12-01'], ly.loc['1959-01-01':]
    m = fit_sarima(tr, (0, 1, 1), (0, 1, 1, 12))
    fc = m.get_forecast(len(te))
    f = np.exp(fc.predicted_mean)
    ci = np.exp(fc.conf_int(alpha=0.05))
    sn = np.exp(np.r_[tr.values[-12:], tr.values[-12:]])
    full = fit_sarima(ly, (0, 1, 1), (0, 1, 1, 12))
    e = full.resid[13:]
    fig, ax = plt.subplots(figsize=(10.4, 3.7))
    ax.plot(y.loc['1955':].index, y.loc['1955':].values, color=st.MainBlue, lw=1.3, label='passengers (thousands)')
    ax.plot(te.index, f, color=st.IDAred, lw=1.4, label='airline model, forecast from Dec 1958')
    ax.fill_between(te.index, ci[:, 0], ci[:, 1], color=st.IDAred, alpha=0.15, label='95% interval')
    ax.plot(te.index, sn, color=st.Amber, lw=1.1, ls='--', label='seasonal naive')
    ax.axvline(pd.Timestamp('1958-12-15'), color=st.DarkText, lw=0.6, ls=':')
    ax.set_title('Airline model: 24 forecasts out of sample')
    st.legend_outside_bottom(ax, ncol=4, y=-0.12)
    plt.tight_layout()
    save('tsa_ch4_airline_fc', save_it)
    act = np.exp(te.values)
    mape = lambda a, b: float(100 * np.mean(np.abs(a - b) / a))
    return {'theta': float(m.params[0]), 'Theta': float(m.params[1]), 'theta_se': float(m.bse[0]), 'Theta_se': float(m.bse[1]),
            'sigma': float(np.sqrt(m.params[2])), 'full_theta': float(full.params[0]), 'full_Theta': float(full.params[1]),
            'full_theta_se': float(full.bse[0]), 'full_Theta_se': float(full.bse[1]), 'full_sigma': float(np.sqrt(full.params[2])),
            'lb24': ljung_box(e, 24, 22), 'mape_sarima': mape(act, f), 'mape_snaive': mape(act, sn),
            'f_last': float(f[-1]), 'lo_last': float(ci[-1, 0]), 'hi_last': float(ci[-1, 1]), 'act_last': float(act[-1]),
            'width_1': float(ci[0, 1] - ci[0, 0]), 'width_24': float(ci[-1, 1] - ci[-1, 0])}


# =============================================================================
# 5. CASE STUDY: ROMANIAN GDP (NOT ADJUSTED)
# =============================================================================
GDP_GRID = [((0, 1, 1), (0, 1, 1)), ((1, 1, 0), (0, 1, 1)), ((1, 1, 1), (0, 1, 1)), ((0, 1, 2), (0, 1, 1)),
            ((2, 1, 0), (0, 1, 1)), ((0, 1, 1), (1, 1, 0)), ((1, 1, 0), (1, 1, 0)), ((0, 1, 1), (1, 1, 1)),
            ((0, 1, 0), (0, 1, 1)), ((0, 1, 1), (0, 1, 0))]


def gdp_models(save_csv=True):
    """SARIMA(p,1,q)(P,1,Q)_4 models of 100 ln GDP (not adjusted), 2000Q1 onwards: AICc, BIC, Ljung-Box (8 lags)."""
    y = 100 * eurostat_log(GDP_NSA, GDP_START)
    rows = []
    n = len(y) - 5
    for o, so in GDP_GRID:
        r = fit_sarima(y, o, so + (4,))
        k = len(r.params) - 1
        lb = ljung_box(r.resid[5:], 8, 8 - k)
        rows.append({'model': f'({o[0]},1,{o[2]})({so[0]},1,{so[2]})4', 'k': k, 'loglik': float(r.llf), 'aicc': aicc(r, n),
                     'bic': float(r.bic), 'lb8_p': lb['lb_p'], 'sigma': float(np.sqrt(r.params[-1])),
                     'params': {nm: float(v) for nm, v in zip(r.param_names, r.params)},
                     'se': {nm: float(v) for nm, v in zip(r.param_names, r.bse)}})
    t = pd.DataFrame(rows)
    if save_csv:
        t.drop(columns=['params', 'se']).round(3).to_csv(os.path.join(HERE, 'ch4_gdp_models.csv'), index=False)
    best_aicc = t.loc[t['aicc'].idxmin(), 'model']
    best_bic = t.loc[t['bic'].idxmin(), 'model']
    return {'rows': rows, 'best_aicc': best_aicc, 'best_bic': best_bic, 'T': int(len(y)), 'first': str(y.index[0].date()),
            'last': str(y.index[-1].date())}


def parse_model(m):
    """'(0,1,1)(0,1,1)4' -> ((0, 1, 1), (0, 1, 1))."""
    a, b = m.split(')(')
    o = tuple(int(x) for x in a.strip('(').split(','))
    so = tuple(int(x) for x in b.split(')')[0].split(','))
    return o, so


def fig_gdp_diag(order=(0, 1, 1), sorder=(0, 1, 1), nlags=16, save_it=True):
    """Residual diagnostics of the GDP SARIMA: residuals, ACF, histogram with the Normal density."""
    y = 100 * eurostat_log(GDP_NSA, GDP_START)
    r = fit_sarima(y, order, sorder + (4,))
    e = pd.Series(r.resid[5:], index=y.index[5:])
    k = sum(order[::2]) + sum(sorder[::2])
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.4))
    ax[0].plot(e.index, e.values, color=st.MainBlue, lw=1.0)
    ax[0].axhline(0, color=st.DarkText, lw=0.5)
    ax[0].set_title('Residuals (%)')
    acf_bars(ax[1], sample_acf(e.values, nlags), len(e), color=st.MainBlue, mark=[4, 8, 12, 16])
    ax[1].set_ylim(-0.6, 0.6)
    ax[1].set_title('ACF of the residuals')
    ax[1].set_xlabel('lag (quarters)')
    ax[2].hist(e.values, bins=22, density=True, color=st.Teal, alpha=0.8, label='residuals')
    g = np.linspace(e.min() - 1, e.max() + 1, 200)
    ax[2].plot(g, stats.norm.pdf(g, e.mean(), e.std()), color=st.IDAred, lw=1.4, label='Normal density')
    ax[2].set_title('Distribution of the residuals')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    plt.tight_layout()
    save('tsa_ch4_gdp_diag', save_it)
    jb = stats.jarque_bera(e.values)
    i = int(np.argmax(np.abs(e.values)))
    ex = e.drop(e.loc['2020'].index)
    return {'lb8': ljung_box(e.values, 8, 8 - k), 'lb12': ljung_box(e.values, 12, 12 - k), 'jb': float(jb.statistic),
            'jb_p': float(jb.pvalue), 'out_d': str(e.index[i].date()), 'out_v': float(e.iloc[i]), 'sd': float(e.std()),
            'out_z': float(e.iloc[i] / e.std()), 'lb8_ex2020': ljung_box(ex.values, 8, 8 - k),
            'jb_ex2020': float(stats.jarque_bera(ex.values).statistic)}


def fig_gdp_forecast(order=(0, 1, 1), sorder=(0, 1, 1), H=8, save_it=True):
    """Forecasts of GDP (not adjusted) 8 quarters ahead with 80% and 95% intervals, back-transformed from 100 ln Y,
    and the implied year-on-year growth."""
    y = 100 * eurostat_log(GDP_NSA, GDP_START)
    r = fit_sarima(y, order, sorder + (4,))
    fc = r.get_forecast(H)
    idx = pd.date_range(y.index[-1] + pd.offsets.QuarterBegin(startingMonth=1), periods=H, freq='QS')
    f = np.exp(fc.predicted_mean / 100) / 1000
    c95 = np.exp(fc.conf_int(alpha=0.05) / 100) / 1000
    c80 = np.exp(fc.conf_int(alpha=0.20) / 100) / 1000
    lev = np.exp(y / 100) / 1000
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.6), gridspec_kw={'width_ratios': [1.6, 1]})
    ax[0].plot(lev.loc['2016':].index, lev.loc['2016':].values, color=st.MainBlue, lw=1.3, label='GDP, not adjusted (bn EUR)')
    ax[0].plot(idx, f, color=st.IDAred, lw=1.4, marker='o', ms=3, label='SARIMA forecast')
    ax[0].fill_between(idx, c95[:, 0], c95[:, 1], color=st.IDAred, alpha=0.12, label='95% interval')
    ax[0].fill_between(idx, c80[:, 0], c80[:, 1], color=st.IDAred, alpha=0.22, label='80% interval')
    ax[0].set_title('Romanian GDP: SARIMA forecasts, 8 quarters')
    st.legend_outside_bottom(ax[0], ncol=2, y=-0.13)
    yy = np.r_[y.values, fc.predicted_mean]
    g = (yy[4:] - yy[:-4])
    gi = pd.date_range(y.index[4], periods=len(g), freq='QS')
    ax[1].bar(gi[-16:-H], g[-16:-H], width=60, color=st.MainBlue, label='observed')
    ax[1].bar(gi[-H:], g[-H:], width=60, color=st.IDAred, label='implied by the forecast')
    ax[1].axhline(0, color=st.DarkText, lw=0.5)
    ax[1].set_title('Year-on-year growth, 100 $\\Delta_4 \\ln Y$ (%)')
    ax[1].xaxis.set_major_locator(mdates.YearLocator())
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    st.legend_outside_bottom(ax[1], ncol=2, y=-0.13)
    plt.tight_layout()
    save('tsa_ch4_gdp_fc', save_it)
    return {'f': f.tolist(), 'lo95': c95[:, 0].tolist(), 'hi95': c95[:, 1].tolist(), 'first_f': str(idx[0].date()),
            'last_f': str(idx[-1].date()), 'g_f': g[-H:].tolist(), 'last_obs': float(lev.iloc[-1]),
            'last_obs_d': str(lev.index[-1].date()), 'params': dict(zip(r.param_names, map(float, r.params))),
            'se': dict(zip(r.param_names, map(float, r.bse))), 'sigma': float(np.sqrt(r.params[-1])),
            'order': [list(order), list(sorder)]}


def fig_sa_nsa(save_it=True):
    """Eurostat seasonally and calendar adjusted (SCA) and unadjusted (NSA) series: GDP and retail trade; the implied
    seasonal-calendar factors NSA/SCA."""
    g, gs = read_eurostat(*GDP_NSA).loc['2010':] / 1000, read_eurostat(*GDP_SCA).loc['2010':] / 1000
    r, rs = read_eurostat(*RETAIL).loc['2015':], read_eurostat(*RETAIL_SCA).loc['2015':]
    fg, fr = (g / gs).dropna(), (r / rs).dropna()
    fig, ax = plt.subplots(2, 2, figsize=(11.0, 5.4))
    ax[0, 0].plot(g.index, g.values, color=st.MainBlue, lw=1.0, label='not adjusted (NSA)')
    ax[0, 0].plot(gs.index, gs.values, color=st.IDAred, lw=1.6, label='adjusted (SCA)')
    ax[0, 0].set_title('GDP, bn EUR (2010 prices)')
    st.legend_outside_bottom(ax[0, 0], ncol=2, y=-0.14)
    ax[0, 1].plot(r.index, r.values, color=st.MainBlue, lw=0.9, label='not adjusted (NSA)')
    ax[0, 1].plot(rs.index, rs.values, color=st.IDAred, lw=1.6, label='adjusted (SCA)')
    st.legend_outside_bottom(ax[0, 1], ncol=2, y=-0.14)
    ax[0, 1].set_title('Retail trade volume, 2021 = 100')
    cols = [st.MainBlue, st.Forest, st.Orange, st.IDAred]
    for q in range(1, 5):
        s = fg[fg.index.quarter == q]
        ax[1, 0].plot(s.index, s.values, color=cols[q - 1], lw=1.3, marker='o', ms=2.5, label=f'Q{q}')
    ax[1, 0].axhline(1, color=st.DarkText, lw=0.5)
    ax[1, 0].set_title('GDP: factor NSA / SCA by quarter')
    st.legend_outside_bottom(ax[1, 0], ncol=4, y=-0.16)
    ax[1, 1].plot(fr.index, fr.values, color=st.Purple, lw=1.0, label='retail: factor NSA / SCA')
    ax[1, 1].axhline(1, color=st.DarkText, lw=0.5)
    ax[1, 1].set_title('Retail trade: factor NSA / SCA')
    st.legend_outside_bottom(ax[1, 1], ncol=1, y=-0.16)
    plt.tight_layout(h_pad=1.5)
    save('tsa_ch4_sa_nsa', save_it)
    fq = fg.loc['2015':'2019'].groupby(fg.loc['2015':'2019'].index.quarter).mean()
    gr_nsa = 100 * np.log(g).diff().loc['2025':]
    gr_sca = 100 * np.log(gs).diff().loc['2025':]
    fm = fr.loc['2016':'2019']
    fm = fm.groupby(fm.index.month).mean()
    return {'fq': fq.round(4).tolist(), 'qoq_nsa': gr_nsa.round(2).tolist(), 'qoq_sca': gr_sca.round(2).tolist(),
            'qoq_dates': [str(d.date()) for d in gr_nsa.index], 'f_dec': float(fm[12]), 'f_jan': float(fm[1]),
            'sum_nsa_2024': float(g.loc['2024'].sum()), 'sum_sca_2024': float(gs.loc['2024'].sum())}


# =============================================================================
# 6. FOURIER TERMS AND CALENDAR EFFECTS
# =============================================================================
def fig_fourier(save_it=True):
    """Fourier approximation of the monthly seasonal pattern of log tourism nights (2012-2019): K = 1, 2, 3 and 6
    harmonics, R^2 of each, and the number of coefficients."""
    x = eurostat_log(TOUR, '2012-01-01').loc[:'2019-12-01']
    z = (x - x.rolling(13, center=True).mean()).dropna()
    pat = z.groupby(z.index.month).mean()
    pat = pat - pat.mean()
    t = np.arange(1, 13)
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.7), gridspec_kw={'width_ratios': [1.5, 1]})
    ax[0].plot(t, pat.values, 'o', color=st.DarkText, ms=6, label='monthly seasonal effect (log)')
    tt = np.linspace(0.6, 12.4, 300)
    out = {}
    for K, c in [(1, st.MainBlue), (2, st.Forest), (3, st.Orange), (6, st.IDAred)]:
        X = fourier(t, 12, K).values
        if K == 6:
            X = X[:, :-1]                 # sin(pi t) = 0 at integer t: 11 regressors
        b = np.linalg.lstsq(X, pat.values, rcond=None)[0]
        Xg = fourier(tt, 12, K).values
        Xg = Xg[:, :-1] if K == 6 else Xg
        fit = X @ b
        r2 = 1 - np.sum((pat.values - fit) ** 2) / np.sum(pat.values ** 2)
        ax[0].plot(tt, Xg @ b, color=c, lw=1.5, label=f'K = {K}: {X.shape[1]} coefficients, $R^2$ = {r2:.3f}')
        out[f'K{K}'] = {'r2': float(r2), 'ncoef': int(X.shape[1])}
    ax[0].set_xticks(range(1, 13))
    ax[0].set_xticklabels(list('JFMAMJJASOND'))
    ax[0].set_title('Tourism nights: seasonal pattern and Fourier fits')
    st.legend_outside_bottom(ax[0], ncol=2, y=-0.13)
    for k, c in [(1, st.MainBlue), (2, st.Forest), (3, st.Orange)]:
        ax[1].plot(tt, np.cos(2 * np.pi * k * tt / 12), color=c, lw=1.3, label=f'$\\cos(2\\pi {k} t/12)$')
        ax[1].plot(tt, np.sin(2 * np.pi * k * tt / 12), color=c, lw=1.0, ls='--', label=f'$\\sin(2\\pi {k} t/12)$')
    ax[1].set_title('The first three harmonics')
    ax[1].set_xticks(range(1, 13))
    st.legend_outside_bottom(ax[1], ncol=2, y=-0.13)
    plt.tight_layout()
    save('tsa_ch4_fourier', save_it)
    out['pat'] = pat.round(4).tolist()
    return out


def easter_regression(key=FOOD, start='2010-01-01', w=10):
    """Regression with SARIMA(0,1,1)(0,1,1)_12 errors for ln(retail volume): Orthodox Easter share (w days),
    working days, additive outliers in April and May 2020 (lockdown)."""
    y = eurostat_log(key, start)
    X = pd.DataFrame({'easter': easter_share(y.index, w), 'wd': working_days(y.index),
                      'ao_2020_04': (y.index == pd.Timestamp('2020-04-01')).astype(float),
                      'ao_2020_05': (y.index == pd.Timestamp('2020-05-01')).astype(float)}, index=y.index)
    r = fit_sarima(y, (0, 1, 1), (0, 1, 1, 12), exog=X.values)
    nm = list(X.columns) + ['theta', 'Theta', 'sigma2']
    return {'params': dict(zip(nm, map(float, r.params))), 'se': dict(zip(nm, map(float, r.bse))),
            'T': int(len(y)), 'first': str(y.index[0].date()), 'last': str(y.index[-1].date()), 'aic': float(r.aic),
            'X': X}


def fig_easter(save_it=True):
    """Orthodox Easter dates and the Easter effect in Romanian retail trade: food retail and total retail."""
    food = easter_regression(FOOD)
    tot = easter_regression(RETAIL)
    yrs = np.arange(2000, 2031)
    ed = [pd.Timestamp(easter(y, EASTER_ORTHODOX)) for y in yrs]
    doy = [d.dayofyear - pd.Timestamp(d.year, 3, 31).dayofyear for d in ed]   # days after 31 March
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.6), gridspec_kw={'width_ratios': [1.3, 1]})
    col = [st.IDAred if d.month == 5 else (st.Forest if d.month == 3 else st.MainBlue) for d in ed]
    ax[0].scatter(yrs, doy, c=col, s=28)
    for m, c in [('April', st.MainBlue), ('May', st.IDAred)]:
        ax[0].scatter([], [], color=c, s=28, label=f'Easter in {m}')
    ax[0].axhline(0.5, color=st.DarkText, lw=0.5)
    ax[0].axhline(30.5, color=st.DarkText, lw=0.5)
    ax[0].set_ylabel('day of April')
    ax[0].set_title('Orthodox Easter Sunday, 2000-2030')
    st.legend_outside_bottom(ax[0], ncol=2, y=-0.13)
    y = eurostat_log(FOOD, '2009-01-01')
    sh = easter_share(y.index)
    g12 = 100 * y.diff(12)
    ds = sh.diff(12)
    pts = pd.DataFrame({'g': g12, 'ds': ds}).dropna()
    pts = pts[pts.index.month.isin([3, 4, 5]) & ~pts.index.year.isin([2020, 2021])]
    for m, c in [(3, st.Forest), (4, st.MainBlue), (5, st.IDAred)]:
        q = pts[pts.index.month == m]
        ax[1].scatter(q['ds'], q['g'], color=c, s=26, label=['March', 'April', 'May'][m - 3])
    b = np.polyfit(pts['ds'], pts['g'], 1)
    xx = np.linspace(-1, 1, 10)
    ax[1].plot(xx, np.polyval(b, xx), color=st.DarkText, lw=1.0, ls='--', label=f'OLS line, slope {b[0]:.1f} pp')
    ax[1].axhline(0, color=st.DarkText, lw=0.4)
    ax[1].axvline(0, color=st.DarkText, lw=0.4)
    ax[1].set_xlabel('change in the Easter share of the month (y/y)')
    ax[1].set_ylabel('y/y growth (%)')
    ax[1].set_title('Food retail: March-May growth and Easter')
    st.legend_outside_bottom(ax[1], ncol=4, y=-0.25)
    plt.tight_layout()
    save('tsa_ch4_easter', save_it)
    keep = lambda d: {k: v for k, v in d.items() if k != 'X'}
    return {'food': keep(food), 'total': keep(tot), 'slope': float(b[0]), 'n_pts': int(len(pts)), 'months_may': [str(y) for y, d in zip(yrs, ed) if d.month == 5 and y <= 2026],
            'easter_dates': {str(y): str(d.date()) for y, d in zip(yrs, ed) if 2022 <= y <= 2027}}


# =============================================================================
# 7. MULTIPLE SEASONALITY: ELECTRICITY LOAD
# =============================================================================
def fig_load_hourly(save_it=True):
    """Hourly load: three weeks in January 2025 and the ACF of one year of hourly data (lags up to 400 hours)."""
    s = load_hourly() / 1000
    w = s.loc['2025-01-13':'2025-02-02 23:00']
    yr = s.loc['2025-01-01':'2025-12-31']
    r = sample_acf(yr.values, 400)
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.6), gridspec_kw={'width_ratios': [1.5, 1]})
    ax[0].plot(w.index, w.values, color=st.Teal, lw=0.9, label='hourly load (GW)')
    for d in pd.date_range('2025-01-18', '2025-02-02', freq='7D'):
        ax[0].axvspan(d, d + pd.Timedelta(days=2), color=st.Amber, alpha=0.15, lw=0)
    ax[0].fill_between([], [], color=st.Amber, alpha=0.15, label='weekends')
    ax[0].set_title('Hourly load, 13 January - 2 February 2025 (GW)')
    date_axis(ax[0], 'week')
    st.legend_outside_bottom(ax[0], ncol=2, y=-0.15)
    ax[1].plot(np.arange(1, 401), r, color=st.Teal, lw=1.0, label='sample ACF, 2025')
    for m in (24, 168, 336):
        ax[1].axvline(m, color=st.Orange, lw=0.8, ls=':')
    ax[1].plot([], [], color=st.Orange, ls=':', label='lags 24, 168, 336')
    ax[1].axhline(0, color=st.DarkText, lw=0.5)
    ax[1].set_xlabel('lag (hours)')
    ax[1].set_title('ACF of hourly load')
    st.legend_outside_bottom(ax[1], ncol=2, y=-0.22)
    plt.tight_layout()
    save('tsa_ch4_load_hourly', save_it)
    prof = yr.groupby(yr.index.hour).mean()
    return {'r12': float(r[11]), 'r24': float(r[23]), 'r168': float(r[167]), 'r84': float(r[83]),
            'peak_h': int(prof.idxmax()), 'trough_h': int(prof.idxmin()), 'peak_v': float(prof.max()),
            'trough_v': float(prof.min()), 'T_year': int(len(yr)), 'T_all': int(len(s)),
            'first': str(s.index[0].date()), 'last': str(s.index[-1].date())}


def fig_mstl(save_it=True):
    """MSTL of hourly load (four weeks, February 2025): trend, daily (24) and weekly (168) components, remainder."""
    s = (load_hourly() / 1000).loc['2025-02-03':'2025-03-02 23:00']
    res = MSTL(s.values, periods=(24, 168)).fit()
    comp = [('load (GW)', s.values, st.Teal), ('trend', res.trend, st.MainBlue),
            ('daily seasonal (period 24)', res.seasonal[:, 0], st.Forest),
            ('weekly seasonal (period 168)', res.seasonal[:, 1], st.Orange), ('remainder', res.resid, st.Purple)]
    fig, ax = plt.subplots(5, 1, figsize=(7.0, 3.7), sharex=True)
    for a, (t, v, c) in zip(ax, comp):
        a.plot(s.index, v, color=c, lw=0.8)
        a.set_ylabel(t, fontsize=10.5, rotation=0, ha='right', va='center')
        a.tick_params(labelsize=10.5)
    ax[0].set_title('MSTL decomposition of hourly load, 3 February - 2 March 2025')
    date_axis(ax[-1], 'week')
    plt.tight_layout()
    save('tsa_ch4_mstl', save_it)
    v = lambda x: float(np.var(x))
    tot = v(s.values - res.trend)
    return {'amp24': float(np.ptp(res.seasonal[:, 0][:24 * 7])), 'amp168': float(np.ptp(res.seasonal[:, 1][:168])),
            'share24': v(res.seasonal[:, 0]) / tot, 'share168': v(res.seasonal[:, 1]) / tot, 'share_r': v(res.resid) / tot}


def fig_load_daily(save_it=True):
    """Daily load 2022-2026 with public holidays marked, and the mean load by day of the week."""
    d = load_daily()
    H = ro_holidays(range(2022, 2027))
    hol = d.reindex(H.index).dropna()
    eas = hol[H.reindex(hol.index) == 'Easter']
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.6), gridspec_kw={'width_ratios': [1.8, 1]})
    ax[0].plot(d.index, d.values, color=st.Teal, lw=0.7, label='daily mean load (GW)')
    ax[0].plot(hol.index, hol.values, 'o', ms=3, color=st.Orange, label='public holidays')
    ax[0].plot(eas.index, eas.values, 'o', ms=4.5, color=st.IDAred, label='Orthodox Easter (Friday to Monday)')
    ax[0].set_title('Daily electricity load in Romania, 2022-2026')
    date_axis(ax[0], 'year')
    st.legend_outside_bottom(ax[0], ncol=3, y=-0.13)
    nh = d[~d.index.isin(H.index)]
    wd = nh.groupby(nh.index.dayofweek).mean()
    ax[1].bar(range(7), wd.values, color=[st.MainBlue] * 5 + [st.Orange, st.IDAred])
    ax[1].set_xticks(range(7))
    ax[1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax[1].set_ylim(wd.min() * 0.85, wd.max() * 1.04)
    ax[1].set_title('Mean load by day (no holidays), GW')
    plt.tight_layout()
    save('tsa_ch4_load_daily', save_it)
    low = d.nsmallest(6)
    return {'wd': wd.round(4).tolist(), 'sun_wed': float(wd[6] / wd[2] - 1), 'sat_wed': float(wd[5] / wd[2] - 1),
            'low_days': [str(x.date()) for x in low.index], 'low_vals': low.round(3).tolist(),
            'easter_mean': float(eas.mean()), 'mean': float(d.mean()), 'n_hol': int(len(hol)),
            'jan_jul': float(d[d.index.month == 1].mean() / d[d.index.month == 5].mean() - 1)}


# =============================================================================
# 8. MODELS FOR DAILY LOAD AND THEIR EVALUATION
# =============================================================================
def load_exog(idx, K=4):
    """Regressors of the dynamic harmonic regression: K annual Fourier pairs, 6 weekday dummies (Sunday is the base),
    Orthodox Easter (Friday to Monday), other public holidays, the Christmas - New Year days (24 Dec - 2 Jan)."""
    X = fourier(idx, 365.25, K)
    for j, nm in enumerate(['mon', 'tue', 'wed', 'thu', 'fri', 'sat']):
        X[nm] = (idx.dayofweek == j).astype(float)
    hol = ro_holidays(range(idx[0].year, idx[-1].year + 1)).reindex(idx)
    X['easter'] = (hol == 'Easter').astype(float)
    X['xmas'] = (((idx.month == 12) & (idx.day >= 24)) | ((idx.month == 1) & (idx.day <= 2))).astype(float)
    X['holiday'] = (hol.notna() & (X['easter'] == 0) & (X['xmas'] == 0)).astype(float)
    return X


def select_orders(y):
    """Orders fixed once, on the data before the first forecast origin: SARIMA(p,0,q)(0,1,1)_7 and the ARMA errors
    of the dynamic harmonic regression, by AIC."""
    out = {}
    best = None
    for o in [(1, 0, 0), (2, 0, 0), (1, 0, 1), (2, 0, 1)]:
        a = fit_sarima(y, o, (0, 1, 1, 7)).aic
        if best is None or a < best[0]:
            best = (a, o)
    out['sarima'] = best[1]
    best = None
    X = load_exog(y.index).values
    for o in [(1, 0, 0), (2, 0, 0), (1, 0, 1), (2, 0, 1), (1, 1, 1)]:
        a = fit_sarima(y, o, (0, 0, 0, 0), exog=X, trend='c' if o[1] == 0 else 'n').aic
        if best is None or a < best[0]:
            best = (a, o)
    out['dhr'] = best[1]
    return out


def forecast_models(tr, h, orders, models=None):
    """h-day forecasts of every model from a training sample `tr` (daily load, GW)."""
    models = models or ['Seasonal naive', 'ETS', 'SARIMA', 'DHR', 'TBATS', 'Prophet']
    idx = pd.date_range(tr.index[-1] + pd.Timedelta(days=1), periods=h, freq='D')
    f = {}
    if 'Seasonal naive' in models:
        f['Seasonal naive'] = np.tile(tr.values[-7:], h // 7 + 1)[:h]
    if 'ETS' in models:
        f['ETS'] = ExponentialSmoothing(tr.values, trend='add', damped_trend=True, seasonal='add',
                                        seasonal_periods=7).fit().forecast(h)
    if 'SARIMA' in models:
        f['SARIMA'] = fit_sarima(tr, orders['sarima'], (0, 1, 1, 7)).forecast(h)
    if 'DHR' in models:
        o = orders['dhr']
        r = fit_sarima(tr, o, (0, 0, 0, 0), exog=load_exog(tr.index).values, trend='c' if o[1] == 0 else 'n')
        f['DHR'] = r.forecast(h, exog=load_exog(idx).values)
    if 'TBATS' in models and have('tbats'):
        from tbats import TBATS
        m = TBATS(seasonal_periods=[7, 365.25], use_box_cox=False, use_trend=False, use_damped_trend=False,
                  use_arma_errors=True, n_jobs=1).fit(tr.values)
        f['TBATS'] = np.asarray(m.forecast(steps=h))
    if 'Prophet' in models and have('prophet'):
        from prophet import Prophet
        H = ro_holidays(range(tr.index[0].year, idx[-1].year + 1))
        p = Prophet(holidays=pd.DataFrame({'ds': H.index, 'holiday': H.values}), daily_seasonality=False,
                    yearly_seasonality=True, weekly_seasonality=True)
        p.fit(pd.DataFrame({'ds': tr.index, 'y': tr.values}))
        f['Prophet'] = p.predict(pd.DataFrame({'ds': idx}))['yhat'].values
    combo = [k for k in f if k != 'Seasonal naive']
    f['Combination'] = np.mean([f[k] for k in combo], axis=0)
    return idx, f


def dm_test(e1, e2, h=1, loss='se'):
    """Diebold-Mariano test of equal accuracy (loss 'se' squared or 'ae' absolute errors), with the Newey-West
    variance on h - 1 lags and the Harvey-Leybourne-Newbold small-sample correction, Student t(n - 1) p-value.
    d_t = L(e1_t) - L(e2_t): a negative mean favours the first forecast."""
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    d = (e1 ** 2 - e2 ** 2) if loss == 'se' else (np.abs(e1) - np.abs(e2))
    n = len(d)
    dbar = d.mean()
    g = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    v = (g[0] + 2 * sum(g[1:])) / n
    dm = dbar / np.sqrt(v)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    hln = corr * dm
    return {'dbar': float(dbar), 'dm': float(dm), 'hln': float(hln), 'p': float(2 * stats.t.sf(abs(hln), n - 1)), 'n': int(n)}


def load_cv(first=CV_FIRST, step=CV_STEP, h=CV_H, n_origins=None, models=None, save_csv=True):
    """Time-series cross-validation on daily load: expanding training window, forecast origins every `step` days from
    `first`, horizon h. Returns the errors of every model at every origin and horizon."""
    y = load_daily()
    orders = select_orders(y.loc[:first])
    origins = [o for o in pd.date_range(first, y.index[-1] - pd.Timedelta(days=h), freq=f'{step}D')]
    if n_origins:
        origins = origins[:n_origins]
    rows = []
    for o in origins:
        idx, f = forecast_models(y.loc[:o], h, orders, models)
        act = y.reindex(idx).values
        for k, v in f.items():
            for j in range(h):
                rows.append({'origin': o, 'model': k, 'h': j + 1, 'date': idx[j], 'actual': act[j], 'fc': v[j],
                             'err': act[j] - v[j]})
    E = pd.DataFrame(rows)
    if save_csv:
        E.round(5).to_csv(os.path.join(HERE, 'ch4_load_cv.csv'), index=False)
    return E, orders


def cv_summary(E, y=None):
    """MAE, RMSE (GW), MASE (scaled by the in-sample MAE of the weekly seasonal naive method on the first training
    sample), by model; Diebold-Mariano tests on window-average losses (one value per origin)."""
    y = load_daily() if y is None else y
    first = E['origin'].min()
    tr = y.loc[:first].values
    scale = np.mean(np.abs(tr[7:] - tr[:-7]))
    g = E.groupby('model')['err']
    tab = pd.DataFrame({'MAE': g.apply(lambda e: np.mean(np.abs(e))), 'RMSE': g.apply(lambda e: np.sqrt(np.mean(e ** 2)))})
    tab['MASE'] = tab['MAE'] / scale
    order = [m for m in ['Seasonal naive', 'ETS', 'SARIMA', 'DHR', 'TBATS', 'Prophet', 'Combination'] if m in tab.index]
    tab = tab.loc[order]
    W = E.assign(se=E['err'] ** 2, ae=E['err'].abs()).groupby(['model', 'origin'])[['se', 'ae']].mean()
    best = tab['MAE'].drop('Combination', errors='ignore').idxmin()
    dm = {}
    for m in order:
        for ref in ['Seasonal naive', 'Combination', best]:
            if m == ref:
                continue
            a, b = W.loc[m], W.loc[ref]
            dm[f'{m}|{ref}|ae'] = dm_raw(a['ae'].values, b['ae'].values)
            dm[f'{m}|{ref}|se'] = dm_raw(a['se'].values, b['se'].values)
    byh = E.assign(ae=E['err'].abs()).groupby(['model', 'h'])['ae'].mean().unstack(0)
    return tab, dm, best, scale, byh


def dm_raw(l1, l2, h=1):
    """Diebold-Mariano test on two loss series (already computed), HLN correction, t(n - 1) p-value."""
    d = np.asarray(l1, float) - np.asarray(l2, float)
    n = len(d)
    dbar = d.mean()
    g = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    v = (g[0] + 2 * sum(g[1:])) / n
    dm = dbar / np.sqrt(v)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) * dm
    return {'dbar': float(dbar), 'dm': float(dm), 'hln': float(hln), 'p': float(2 * stats.t.sf(abs(hln), n - 1)), 'n': int(n)}


def fig_cv_scheme(save_it=True, n=6):
    """The expanding-window (rolling-origin) scheme of time-series cross-validation."""
    fig, ax = plt.subplots(figsize=(10.0, 3.0))
    for i in range(n):
        tr_end = 10 + 2 * i
        ax.barh(i, tr_end, left=0, color=st.MainBlue, height=0.55, label='training sample' if i == 0 else '_nolegend_')
        ax.barh(i, 3, left=tr_end, color=st.IDAred, height=0.55, label='forecast window (h = 3)' if i == 0 else '_nolegend_')
        ax.barh(i, 26 - tr_end - 3, left=tr_end + 3, color=st.Amber, alpha=0.25, height=0.55,
                label='not used at this origin' if i == 0 else '_nolegend_')
        ax.text(-0.4, i, f'origin {i + 1}', ha='right', va='center', fontsize=10, color=st.DarkText)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlim(-4, 26)
    ax.set_xlabel('time')
    ax.spines['left'].set_visible(False)
    ax.set_title('Time-series cross-validation: the origin moves forward, the model is refitted')
    st.legend_outside_bottom(ax, ncol=3, y=-0.3)
    plt.tight_layout()
    save('tsa_ch4_cv_scheme', save_it)
    return {'n': n}


def fig_load_forecasts(E, origin=None, save_it=True):
    """Forecasts of the models from one origin against the actual daily load (the window with Easter if present)."""
    y = load_daily()
    if origin is None:
        eas = pd.Timestamp(easter(2026, EASTER_ORTHODOX))
        cand = [o for o in sorted(E['origin'].unique()) if o < eas <= o + pd.Timedelta(days=CV_H)]
        origin = cand[0] if cand else sorted(E['origin'].unique())[-1]
    sub = E[E['origin'] == origin]
    fig, ax = plt.subplots(figsize=(10.6, 3.8))
    hist = y.loc[origin - pd.Timedelta(days=27):origin + pd.Timedelta(days=CV_H)]
    ax.plot(hist.index, hist.values, color=st.DarkText, lw=1.6, marker='o', ms=2.5, label='actual load')
    for m in ['Seasonal naive', 'SARIMA', 'DHR', 'TBATS', 'Prophet', 'Combination']:
        s = sub[sub['model'] == m]
        if len(s):
            ax.plot(s['date'], s['fc'], color=MODEL_COL[m], lw=1.3, ls='--' if m == 'Seasonal naive' else '-', label=m)
    ax.axvline(origin + pd.Timedelta(hours=12), color=st.DarkText, lw=0.6, ls=':')
    ax.set_title(f'Daily load (GW): forecasts made on {origin.strftime("%d %B %Y")}, 14 days')
    date_axis(ax, 'week')
    st.legend_outside_bottom(ax, ncol=4, y=-0.13)
    plt.tight_layout()
    save('tsa_ch4_load_fc', save_it)
    acc = sub.assign(ae=sub['err'].abs()).groupby('model')['ae'].mean()
    return {'origin': str(pd.Timestamp(origin).date()), 'mae': acc.round(4).to_dict()}


def fig_cv_results(tab, byh, save_it=True):
    """MAE by model (bars) and MAE by horizon (lines)."""
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.7), gridspec_kw={'width_ratios': [1, 1.3]})
    ax[0].barh(range(len(tab)), tab['MAE'].values * 1000, color=[MODEL_COL[m] for m in tab.index])
    ax[0].set_yticks(range(len(tab)))
    ax[0].set_yticklabels(tab.index)
    ax[0].invert_yaxis()
    ax[0].set_xlabel('MAE (MW)')
    ax[0].set_title('Mean absolute error, all origins and horizons')
    for i, v in enumerate(tab['MAE'].values * 1000):
        ax[0].text(v + 4, i, f'{v:.0f}', va='center', fontsize=9.5, color=st.DarkText)
    for m in byh.columns:
        ax[1].plot(byh.index, byh[m].values * 1000, color=MODEL_COL[m], lw=1.8 if m == 'Combination' else 1.2,
                   ls='--' if m == 'Seasonal naive' else '-', marker='o', ms=2.5, label=m)
    ax[1].set_xlabel('horizon h (days)')
    ax[1].set_ylabel('MAE (MW)')
    ax[1].set_title('MAE by forecast horizon')
    st.legend_outside_bottom(ax[1], ncol=4, y=-0.2)
    plt.tight_layout()
    save('tsa_ch4_cv_results', save_it)


def fig_prophet_components(save_it=True, end=CV_FIRST):
    """Prophet on daily load up to the first forecast origin: trend, weekly and yearly components, holiday effects."""
    if not have('prophet'):
        return {}
    from prophet import Prophet
    y = load_daily().loc[:end]
    H = ro_holidays(range(2022, 2027))
    p = Prophet(holidays=pd.DataFrame({'ds': H.index, 'holiday': H.values}), daily_seasonality=False)
    p.fit(pd.DataFrame({'ds': y.index, 'y': y.values}))
    fut = p.make_future_dataframe(periods=0)
    c = p.predict(fut)
    fig, ax = plt.subplots(2, 2, figsize=(11.0, 5.0))
    ax[0, 0].plot(y.index, y.values, color=st.Teal, lw=0.5, label='daily load')
    ax[0, 0].plot(c['ds'], c['trend'], color=st.MainBlue, lw=1.8, label='trend $g(t)$')
    for cp in p.changepoints[np.abs(np.nanmean(p.params['delta'], axis=0)) > 0.01]:
        ax[0, 0].axvline(cp, color=st.Orange, lw=0.6, ls=':')
    ax[0, 0].plot([], [], color=st.Orange, ls=':', label='changepoints used')
    ax[0, 0].set_title('Trend (GW)')
    date_axis(ax[0, 0], 'year')
    st.legend_outside_bottom(ax[0, 0], ncol=3, y=-0.15)
    wk = c.groupby(c['ds'].dt.dayofweek)['weekly'].mean()
    ax[0, 1].bar(range(7), wk.values, color=[st.MainBlue] * 5 + [st.Orange, st.IDAred])
    ax[0, 1].set_xticks(range(7))
    ax[0, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax[0, 1].axhline(0, color=st.DarkText, lw=0.5)
    ax[0, 1].set_title('Weekly component (GW)')
    yr = c[(c['ds'] >= '2024-01-01') & (c['ds'] <= '2024-12-31')]
    ax[1, 0].plot(yr['ds'], yr['yearly'], color=st.Forest, lw=1.5)
    ax[1, 0].axhline(0, color=st.DarkText, lw=0.5)
    ax[1, 0].set_title('Yearly component, shown for 2024 (GW)')
    date_axis(ax[1, 0], 'month')
    hn = [h for h in p.train_holiday_names]
    eff = pd.Series({h: c.loc[c[h] != 0, h].mean() if (c[h] != 0).any() else 0.0 for h in hn}).sort_values()
    ax[1, 1].barh(range(len(eff)), eff.values, color=[st.IDAred if v < -0.6 else st.Orange for v in eff.values])
    ax[1, 1].set_yticks(range(len(eff)))
    ax[1, 1].set_yticklabels(eff.index, fontsize=9)
    ax[1, 1].axvline(0, color=st.DarkText, lw=0.5)
    ax[1, 1].set_title('Holiday effects (GW)')
    plt.tight_layout()
    save('tsa_ch4_prophet', save_it)
    return {'weekly': wk.round(4).tolist(), 'yearly_max': float(yr['yearly'].max()), 'yearly_min': float(yr['yearly'].min()),
            'yearly_argmax': str(yr.loc[yr['yearly'].idxmax(), 'ds'].date()), 'yearly_argmin': str(yr.loc[yr['yearly'].idxmin(), 'ds'].date()),
            'hol': eff.round(4).to_dict(), 'trend0': float(c['trend'].iloc[0]), 'trend1': float(c['trend'].iloc[-1]),
            'n_cp': int(len(p.changepoints))}


def tbats_fit(end=CV_FIRST):
    """TBATS(7, 365.25) on daily load up to the first origin: chosen harmonics, ARMA orders, AIC."""
    if not have('tbats'):
        return {}
    from tbats import TBATS
    y = load_daily().loc[:end]
    m = TBATS(seasonal_periods=[7, 365.25], use_box_cox=False, use_trend=False, use_damped_trend=False,
              use_arma_errors=True, n_jobs=1).fit(y.values)
    p = m.params
    return {'harmonics': [int(k) for k in p.components.seasonal_harmonics], 'p': int(len(p.ar_coefs)),
            'q': int(len(p.ma_coefs)), 'aic': float(m.aic), 'alpha': float(p.alpha),
            'n_states': int(1 + 2 * sum(p.components.seasonal_harmonics) + len(p.ar_coefs) + len(p.ma_coefs))}


def dhr_fit(end=CV_FIRST, order=None):
    """The dynamic harmonic regression on daily load up to the first origin: holiday and weekday coefficients."""
    y = load_daily().loc[:end]
    order = order or select_orders(y)['dhr']
    X = load_exog(y.index)
    r = fit_sarima(y, order, (0, 0, 0, 0), exog=X.values, trend='c' if order[1] == 0 else 'n')
    nm = (['const'] if order[1] == 0 else []) + list(X.columns)
    par = dict(zip(r.param_names, map(float, r.params)))
    se = dict(zip(r.param_names, map(float, r.bse)))
    keys = [k for k in r.param_names if k.startswith('x')]
    named = {nm_: (par[k], se[k]) for nm_, k in zip(X.columns, keys)}
    e = r.resid[10:]
    return {'order': list(order), 'coef': {k: v[0] for k, v in named.items()}, 'se': {k: v[1] for k, v in named.items()},
            'ar': [par[k] for k in r.param_names if k.startswith('ar.')], 'ma': [par[k] for k in r.param_names if k.startswith('ma.')],
            'sigma': float(np.sqrt(par['sigma2'])), 'lb14': ljung_box(e, 14, 14 - sum(order[::2])), 'k': int(len(r.params)),
            'aic': float(r.aic)}


def fig_combination(E, m1='DHR', m2='Prophet', save_it=True):
    """MAE of the combination w f1 + (1 - w) f2 as a function of the weight w, on the cross-validation errors."""
    a = E[E['model'] == m1].sort_values(['origin', 'h'])
    b = E[E['model'] == m2].sort_values(['origin', 'h'])
    act = a['actual'].values
    w = np.linspace(0, 1, 101)
    mae = [np.mean(np.abs(act - (x * a['fc'].values + (1 - x) * b['fc'].values))) * 1000 for x in w]
    rm = [np.sqrt(np.mean((act - (x * a['fc'].values + (1 - x) * b['fc'].values)) ** 2)) * 1000 for x in w]
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.plot(w, mae, color=st.MainBlue, lw=1.6, label='MAE (MW)')
    ax.plot(w, rm, color=st.IDAred, lw=1.6, label='RMSE (MW)')
    ax.axvline(0.5, color=st.Forest, lw=0.9, ls='--', label='equal weights')
    ax.set_xlabel(f'weight w of {m1} (1 - w on {m2})')
    ax.set_title(f'Combining {m1} and {m2}: error as a function of the weight')
    st.legend_outside_bottom(ax, ncol=3, y=-0.25)
    plt.tight_layout()
    save('tsa_ch4_combination', save_it)
    e1, e2 = act - a['fc'].values, act - b['fc'].values
    return {'mae_w0': mae[0], 'mae_w1': mae[-1], 'mae_half': mae[50], 'w_best': float(w[int(np.argmin(mae))]),
            'mae_best': float(min(mae)), 'rmse_half': rm[50], 'rmse_w0': rm[0], 'rmse_w1': rm[-1],
            'corr': float(np.corrcoef(e1, e2)[0, 1]), 'sd1': float(e1.std() * 1000), 'sd2': float(e2.std() * 1000)}


# =============================================================================
# 9. ROLLING-ORIGIN EVALUATION FOR QUARTERLY GDP
# =============================================================================
def gdp_cv(first='2012-10-01', H=4):
    """Rolling-origin forecasts of 100 ln GDP (not adjusted), expanding window: SARIMA airline (0,1,1)(0,1,1)_4,
    ETS (additive damped trend, additive seasonality) on the log, seasonal naive; errors in % (log points)."""
    y = 100 * eurostat_log(GDP_NSA, GDP_START)
    rows = []
    for i in range(len(y)):
        o = y.index[i]
        if o < pd.Timestamp(first) or i + H >= len(y):
            continue
        tr = y.iloc[:i + 1]
        act = y.iloc[i + 1:i + 1 + H].values
        f = {'SARIMA': fit_sarima(tr, (0, 1, 1), (0, 1, 1, 4)).forecast(H),
             'ETS': ExponentialSmoothing(tr.values, trend='add', damped_trend=True, seasonal='add', seasonal_periods=4).fit().forecast(H),
             'Seasonal naive': tr.values[-4:][:H] + 0.0}
        f['Combination'] = (f['SARIMA'] + f['ETS']) / 2
        for k, v in f.items():
            for j in range(H):
                rows.append({'origin': o, 'model': k, 'h': j + 1, 'err': act[j] - v[j]})
    E = pd.DataFrame(rows)
    out = {'first': str(E['origin'].min().date()), 'last': str(E['origin'].max().date()), 'n': int(E['origin'].nunique())}
    tab = {}
    for (m, h), g in E.groupby(['model', 'h']):
        tab.setdefault(m, {})[int(h)] = {'mae': float(g['err'].abs().mean()), 'rmse': float(np.sqrt((g['err'] ** 2).mean()))}
    out['tab'] = tab
    ex = E[~E['origin'].between('2019-01-01', '2020-12-31')]
    out['tab_ex'] = {m: {int(h): float(g['err'].abs().mean()) for h, g in gm.groupby('h')} for m, gm in ex.groupby('model')}
    out['n_ex'] = int(ex['origin'].nunique())
    dm = {}
    for h in (1, 4):
        a = E[(E['model'] == 'SARIMA') & (E['h'] == h)].sort_values('origin')['err'].values
        for ref in ['ETS', 'Seasonal naive']:
            b = E[(E['model'] == ref) & (E['h'] == h)].sort_values('origin')['err'].values
            dm[f'h{h}|{ref}'] = dm_test(a, b, h=h, loss='se')
    out['dm'] = dm
    return out


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    st.apply()
    N = {}
    for name, f in [('series', fig_series), ('splot', fig_seasonal_plot), ('circle', fig_unit_circle),
                    ('gdpdiff', fig_gdp_differencing), ('tests', seasonal_tests), ('theory', fig_theory_acf),
                    ('air', fig_airline), ('airfc', fig_airline_forecast), ('gdpm', gdp_models), ('sa', fig_sa_nsa), ('fourier', fig_fourier), ('easter', fig_easter),
                    ('hourly', fig_load_hourly), ('mstl', fig_mstl), ('daily', fig_load_daily), ('cvs', fig_cv_scheme),
                    ('prophet', fig_prophet_components), ('tbats', tbats_fit), ('dhr', dhr_fit), ('gdpcv', gdp_cv)]:
        print(name, flush=True)
        N[name] = f()
    o, so = parse_model(N['gdpm']['best_bic'])
    N['gdpdiag'] = fig_gdp_diag(o, so)
    N['gdpfc'] = fig_gdp_forecast(o, so)
    print('load cv (TBATS at every origin: several minutes)', flush=True)
    E, orders = load_cv()
    tab, dm, best, scale, byh = cv_summary(E)
    fig_cv_results(tab, byh)
    N['loadfc'] = fig_load_forecasts(E)
    N['comb'] = fig_combination(E, 'DHR', 'Prophet')
    N['cv'] = {'tab': tab.round(5).to_dict(orient='index'), 'dm': dm, 'best': best, 'scale': float(scale),
               'orders': {k: list(v) for k, v in orders.items()}, 'n_origins': int(E['origin'].nunique()),
               'first_origin': str(E['origin'].min().date()), 'last_origin': str(E['origin'].max().date()),
               'byh': byh.round(5).to_dict(), 'last_date': str(E['date'].max().date())}
    with open(os.path.join(HERE, 'ch4_numbers.json'), 'w') as fh:
        json.dump(N, fh, indent=1, default=str)
    print('written ch4_numbers.json')
