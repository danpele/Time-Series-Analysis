"""
generate_all_charts.py -- charts and numbers of Chapter 0 (TSA): Introduction, components and exponential smoothing
===================================================================================================================
Every chart and every number shown in the Chapter 0 lecture, from public data and the course data (tsa_data.py),
with the course chart style (tsa_style.py):
  * fig_gdp          -- Romanian real GDP, quarterly, unadjusted and seasonally adjusted (Eurostat);
  * fig_hicp         -- Romanian harmonised consumer price index and its annual inflation rate (Eurostat);
  * fig_eurron       -- the EUR/RON reference rate of the BNR, daily;
  * fig_markets      -- BET and S&P 500, value of 100 invested in January 2000 (log scale);
  * fig_electricity  -- net electricity generation in Romania, monthly (Eurostat);
  * fig_co2          -- atmospheric CO2 at Mauna Loa, monthly averages (statsmodels data set);
  * fig_slutsky      -- Slutsky's experiment: moving sums of white noise look like cycles;
  * fig_components   -- classical multiplicative decomposition of the Romanian GDP (log scale, period 4);
  * fig_stl          -- STL decomposition of the CO2 series (period 12);
  * fig_returns      -- BET: closing price and daily log returns;
  * fig_acf          -- sample ACF of three series: CO2 level, electricity generation, BET daily log returns;
  * fig_ses          -- simple exponential smoothing of the monthly EUR/RON with two smoothing constants;
  * fig_forecast     -- electricity: naive, seasonal naive and Holt-Winters forecasts for the last 24 months;
  * fig_benchmarks   -- MASE of four methods on four seasonal series (a small M-competition on Romanian data).
Output: charts/tsa_ch0_*.pdf/.png, Quantlets/Ch_00/ch0_values.json, Quantlets/Ch_00/ch0_accuracy.csv
Run:  python3 Quantlets/Ch_00/generate_all_charts.py
Time Series Analysis - Daniel Traian PELE
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
from tsa_data import load_close, log_returns, read_eurostat, load_statsmodels   # noqa: E402
import tsa_style as st                                                         # noqa: E402

TABLE_DIR = HERE
# Eurostat series keys (public SDMX API, no key)
GDP_NSA = ('namq_10_gdp', 'Q.CLV10_MEUR.NSA.B1GQ.RO')      # real GDP, chain-linked 2010 EUR, unadjusted
GDP_SCA = ('namq_10_gdp', 'Q.CLV10_MEUR.SCA.B1GQ.RO')      # the same, seasonally and calendar adjusted
HICP = ('prc_hicp_minr', 'M.I15.TOTAL.RO')                  # HICP, all items, 2015 = 100
ELEC = ('nrg_cb_pem', 'M.TOTAL.GWH.RO')                     # net electricity generation, GWh
INFL_START = '2005-01-01'                                   # inflation panel (1997: above 150%)
TEST_H = {'gdp': 8, 'hicp': 24, 'elec': 24, 'co2': 24}      # test-set length of the forecast comparison
SEASON = {'gdp': 4, 'hicp': 12, 'elec': 12, 'co2': 12}
SERIES_LABEL = {'gdp': 'GDP (quarterly)', 'hicp': 'HICP (monthly)', 'elec': 'Electricity (monthly)',
                'co2': 'CO2 (monthly)'}
VALUES = {}


def gdp_series():
    """Romanian real GDP (million EUR, chain-linked 2010 prices): unadjusted and seasonally adjusted."""
    nsa = read_eurostat(*GDP_NSA).rename('nsa')
    sca = read_eurostat(*GDP_SCA).rename('sca')
    df = pd.concat([nsa, sca], axis=1).dropna()
    df.index = pd.PeriodIndex(df.index, freq='Q').to_timestamp()
    return df


def hicp_series():
    """Romanian HICP (2015 = 100), monthly, and the annual inflation rate in %."""
    h = read_eurostat(*HICP).rename('hicp')
    h.index = pd.PeriodIndex(h.index, freq='M').to_timestamp()
    infl = (100 * (h / h.shift(12) - 1)).rename('inflation')
    return pd.concat([h, infl], axis=1)


def electricity_series():
    """Net electricity generation in Romania, GWh per month."""
    e = read_eurostat(*ELEC).rename('elec')
    e.index = pd.PeriodIndex(e.index, freq='M').to_timestamp()
    return e


def co2_series():
    """Mauna Loa CO2 (ppm): weekly data of statsmodels averaged by month, gaps filled by linear interpolation."""
    c = load_statsmodels('co2').resample('MS').mean().interpolate()
    return c.rename('co2')


def fig_gdp(save=True):
    df = gdp_series()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(df.index, df['nsa'] / 1000, color=st.MainBlue, lw=1.2, label='Unadjusted (as measured)')
    ax.plot(df.index, df['sca'] / 1000, color=st.IDAred, lw=1.8, label='Seasonally and calendar adjusted')
    ax.set_ylabel('Billion EUR (2010 prices)')
    st.legend_outside_bottom(ax, ncol=2, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('tsa_ch0_gdp')
    return df


def fig_hicp(save=True):
    df = hicp_series()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].plot(df.index, df['hicp'], color=st.MainBlue, lw=1.4, label='HICP, 2015 = 100')
    axes[0].set_ylabel('Index (log scale)')
    axes[0].set_yscale('log')
    inf = df['inflation'].loc[INFL_START:].dropna()
    axes[1].plot(inf.index, inf, color=st.IDAred, lw=1.4, label=f'Annual inflation rate since {INFL_START[:4]}, %')
    axes[1].axhline(2.5, color=st.Forest, ls='--', lw=1, label='BNR target 2.5%')
    axes[1].axhline(0, color=st.DarkText, lw=0.5)
    axes[1].set_ylabel('%')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('tsa_ch0_hicp')
    return df


def fig_eurron(save=True):
    s = load_close('eurron')
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(s.index, s, color=st.Forest, lw=1.1, label='EUR/RON, BNR reference rate (lei per euro)')
    ax.set_ylabel('RON per EUR')
    st.legend_outside_bottom(ax, ncol=1, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('tsa_ch0_eurron')
    return s


def fig_markets(save=True):
    px = pd.concat([load_close('bet', '2000-01-01'), load_close('sp500', '2000-01-01')], axis=1).ffill().dropna()
    idx = 100 * px / px.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(idx.index, idx['bet'], color=st.COL['bet'], lw=1.2, label='BET (Bucharest Stock Exchange)')
    ax.plot(idx.index, idx['sp500'], color=st.COL['sp500'], lw=1.2, label='S&P 500')
    ax.set_yscale('log')
    ax.set_ylabel('Value of 100 invested (log scale)')
    st.legend_outside_bottom(ax, ncol=2, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('tsa_ch0_markets')
    return idx


def fig_electricity(save=True):
    e = electricity_series()
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(e.index, e / 1000, color=st.Orange, lw=1.3, marker='o', ms=2.2, label='Net electricity generation, Romania')
    jan = e[e.index.month == 1]
    ax.scatter(jan.index, jan / 1000, color=st.MainBlue, s=18, zorder=3, label='January')
    ax.set_ylabel('TWh per month')
    st.legend_outside_bottom(ax, ncol=2, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('tsa_ch0_electricity')
    return e


def fig_co2(save=True):
    c = co2_series()
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(c.index, c, color=st.Purple, lw=1.1, label='CO2 concentration at Mauna Loa, monthly mean (ppm)')
    ax.set_ylabel('ppm')
    st.legend_outside_bottom(ax, ncol=1, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('tsa_ch0_co2')
    return c


def fig_slutsky(n=240, k=10, seed=7, save=True):
    """Slutsky (1937): a moving sum of k independent shocks produces smooth, cycle-like waves."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n + k - 1)
    ma = np.convolve(eps, np.ones(k), mode='valid')            # y_t = eps_t + ... + eps_{t-k+1}
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
    axes[0].plot(np.arange(n), eps[k - 1:], color=st.MainBlue, lw=0.9, label='White noise: independent shocks')
    axes[1].plot(np.arange(n), ma, color=st.IDAred, lw=1.3, label=f'Moving sum of {k} consecutive shocks')
    for a in axes:
        a.axhline(0, color=st.DarkText, lw=0.5)
    axes[1].set_xlabel('t')
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('tsa_ch0_slutsky')
    # number of up-crossings of zero: about one wave every 2k periods
    up = int(((ma[:-1] < 0) & (ma[1:] >= 0)).sum())
    return eps, ma, up


def classical_decomposition(y, m, model='multiplicative'):
    """Classical decomposition: centred moving average of order m (2 x m for even m), seasonal indices as the
    average detrended value of each season (normalised), remainder = what is left."""
    if m % 2 == 0:
        w = np.r_[0.5, np.ones(m - 1), 0.5] / m                # 2 x m centred moving average
    else:
        w = np.ones(m) / m
    trend = pd.Series(np.convolve(y.values, w, mode='same'), index=y.index)
    half = len(w) // 2
    trend.iloc[:half] = np.nan
    trend.iloc[-half:] = np.nan
    detr = y / trend if model == 'multiplicative' else y - trend
    pos = np.arange(len(y)) % m
    raw = pd.Series(detr.values).groupby(pos).mean()
    idx = raw / raw.mean() if model == 'multiplicative' else raw - raw.mean()
    seasonal = pd.Series(idx.values[pos], index=y.index)
    rem = y / (trend * seasonal) if model == 'multiplicative' else y - trend - seasonal
    season_of = {p: y.index[p] for p in range(m)}
    return trend, seasonal, rem, idx, season_of


def fig_components(save=True):
    y = gdp_series()['nsa'] / 1000
    trend, seas, rem, idx, _ = classical_decomposition(y, 4, 'multiplicative')
    fig, axes = plt.subplots(4, 1, figsize=(10, 6.4), sharex=True)
    axes[0].plot(y.index, y, color=st.MainBlue, lw=1.1, label='Observed $Y_t$ (billion EUR)')
    axes[0].plot(trend.index, trend, color=st.IDAred, lw=1.6, label='Trend-cycle $T_t$ (2x4 moving average)')
    axes[1].plot(trend.index, trend, color=st.IDAred, lw=1.6)
    axes[1].set_ylabel('$T_t$')
    axes[2].plot(seas.index, seas, color=st.Forest, lw=1.1, label='Seasonal factor $S_t$')
    axes[2].set_ylabel('$S_t$')
    axes[3].plot(rem.index, rem, color=st.Purple, lw=1.0, label='Remainder $R_t$')
    axes[3].axhline(1, color=st.DarkText, lw=0.5)
    axes[3].set_ylabel('$R_t$')
    axes[0].set_ylabel('$Y_t$')
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('tsa_ch0_components')
    q = {int(y.index[p].quarter): float(idx[p]) for p in range(4)}
    return y, trend, seas, rem, q


def fig_stl(save=True):
    from statsmodels.tsa.seasonal import STL
    c = co2_series()
    res = STL(c, period=12, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(10, 6.4), sharex=True)
    axes[0].plot(c.index, c, color=st.Purple, lw=1.0, label='Observed CO2 (ppm)')
    axes[1].plot(c.index, res.trend, color=st.IDAred, lw=1.4, label='Trend (LOESS)')
    axes[2].plot(c.index, res.seasonal, color=st.Forest, lw=0.9, label='Seasonal (changes slowly)')
    axes[3].plot(c.index, res.resid, color=st.MainBlue, lw=0.8, label='Remainder')
    for a, lab in zip(axes, ['$Y_t$', '$T_t$', '$S_t$', '$R_t$']):
        a.set_ylabel(lab)
    st.fig_legend_bottom(fig, ncol=4, y=0.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('tsa_ch0_stl')
    return c, res


def fig_returns(save=True):
    p = load_close('bet', '2000-01-01')
    r = log_returns('bet', '2000-01-01')
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True)
    axes[0].plot(p.index, p, color=st.COL['bet'], lw=1.0, label='BET, closing value $P_t$ (points)')
    axes[1].plot(r.index, r, color=st.MainBlue, lw=0.5, label='Daily log return $r_t = 100(\\ln P_t - \\ln P_{t-1})$, %')
    axes[0].set_ylabel('Points')
    axes[1].set_ylabel('%')
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('tsa_ch0_returns')
    return p, r


def sample_acf(x, nlags):
    """Sample autocorrelations r_1..r_nlags: sum of (x_t - mean)(x_{t-k} - mean) over T, divided by the variance."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    c0 = np.sum(x * x) / len(x)
    return np.array([np.sum(x[k:] * x[:-k]) / len(x) / c0 for k in range(1, nlags + 1)])


def fig_acf(save=True):
    c = co2_series()
    e = electricity_series()
    r = log_returns('bet', '2000-01-01')
    panels = [('CO2 level (monthly)', c, 36, st.Purple), ('Electricity generation (monthly)', e, 36, st.Orange),
              ('BET daily log returns', r, 36, st.MainBlue)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    out = {}
    for ax, (lab, x, L, col) in zip(axes, panels):
        a = sample_acf(x, L)
        band = 1.96 / np.sqrt(len(x))
        ax.vlines(np.arange(1, L + 1), 0, a, color=col, lw=2, label=lab)
        ax.axhline(0, color=st.DarkText, lw=0.6)
        ax.axhspan(-band, band, color=st.MainBlue, alpha=0.12, lw=0)
        ax.set_xlabel('Lag $k$')
        ax.set_ylim(-0.6, 1.05)
        out[lab] = (a, band, len(x))
    axes[0].set_ylabel('Sample autocorrelation $r_k$')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    st.check_no_grey(fig)
    if save:
        st.save_fig('tsa_ch0_acf')
    return out


def ses_path(y, alpha, level0=None):
    """Simple exponential smoothing: level l_t = alpha y_t + (1 - alpha) l_{t-1}; the forecast of y_{t+1} is l_t."""
    lev = np.empty(len(y))
    prev = y.iloc[0] if level0 is None else level0
    for i, v in enumerate(y.values):
        prev = alpha * v + (1 - alpha) * prev
        lev[i] = prev
    return pd.Series(lev, index=y.index)


def eurron_monthly(start='2015-01-01'):
    s = load_close('eurron', start)
    return s.resample('MS').mean().rename('eurron')


def fig_ses(save=True):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    y = eurron_monthly()
    fit = SimpleExpSmoothing(y, initialization_method='estimated').fit()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(y.index, y, color=st.DarkText, lw=0.0, marker='o', ms=3, label='EUR/RON, monthly average')
    ax.plot(y.index, ses_path(y, 0.1).shift(1), color=st.MainBlue, lw=1.6, label='SES, $\\alpha = 0.1$')
    ax.plot(y.index, ses_path(y, 0.7).shift(1), color=st.IDAred, lw=1.4, label='SES, $\\alpha = 0.7$')
    ax.set_ylabel('RON per EUR')
    st.legend_outside_bottom(ax, ncol=3, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('tsa_ch0_ses')
    return y, float(fit.params['smoothing_level'])


def method_forecasts(y, m, h):
    """Forecasts of the last h values from the first len(y) - h: naive, seasonal naive, SES and Holt-Winters
    (additive damped trend, multiplicative seasonality), all fitted on the training set only."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
    train, test = y.iloc[:-h], y.iloc[-h:]
    f = {}
    f['Naive'] = pd.Series(train.iloc[-1], index=test.index)
    f['Seasonal naive'] = pd.Series([train.iloc[len(train) - m + (i % m)] for i in range(h)], index=test.index)
    f['SES'] = pd.Series(SimpleExpSmoothing(train, initialization_method='estimated').fit().forecast(h).values,
                         index=test.index)
    hw = ExponentialSmoothing(train, trend='add', damped_trend=True, seasonal='mul', seasonal_periods=m,
                              initialization_method='estimated').fit()
    f['Holt-Winters'] = pd.Series(hw.forecast(h).values, index=test.index)
    return train, test, f


def accuracy(train, test, fc, m):
    """MAE, RMSE, MAPE (%) and MASE; the MASE scale is the in-sample MAE of the seasonal naive method (lag m)."""
    e = test - fc
    scale = np.mean(np.abs(train.values[m:] - train.values[:-m]))
    return {'MAE': float(np.mean(np.abs(e))), 'RMSE': float(np.sqrt(np.mean(e ** 2))),
            'MAPE': float(100 * np.mean(np.abs(e / test))), 'MASE': float(np.mean(np.abs(e)) / scale)}


def fig_forecast(save=True):
    y = electricity_series() / 1000
    train, test, f = method_forecasts(y, 12, TEST_H['elec'])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    tail = train.iloc[-48:]
    ax.plot(tail.index, tail, color=st.MainBlue, lw=1.3, label='Training data')
    ax.plot(test.index, test, color=st.DarkText, lw=1.8, label='Test data (actual)')
    cols = {'Naive': st.Amber, 'Seasonal naive': st.Forest, 'Holt-Winters': st.IDAred}
    for k, c in cols.items():
        ax.plot(f[k].index, f[k], color=c, lw=1.5, ls='--', label=k)
    ax.axvline(test.index[0], color=st.Purple, lw=0.8, ls=':')
    ax.set_ylabel('TWh per month')
    st.legend_outside_bottom(ax, ncol=5, y=-0.12)
    st.check_no_grey(fig)
    if save:
        st.save_fig('tsa_ch0_forecast')
    acc = {k: accuracy(train, test, v, 12) for k, v in f.items()}
    return train, test, f, acc


def benchmark_data():
    return {'gdp': gdp_series()['nsa'] / 1000, 'hicp': hicp_series()['hicp'],
            'elec': electricity_series() / 1000, 'co2': co2_series()}


def fig_benchmarks(save=True):
    data = benchmark_data()
    rows = []
    for k, y in data.items():
        train, test, f = method_forecasts(y, SEASON[k], TEST_H[k])
        for meth, fc in f.items():
            a = accuracy(train, test, fc, SEASON[k])
            rows.append(dict(series=k, method=meth, **a))
    tab = pd.DataFrame(rows)
    piv = tab.pivot(index='series', columns='method', values='MASE').loc[list(data)]
    meths = ['Naive', 'Seasonal naive', 'SES', 'Holt-Winters']
    cols = [st.Amber, st.Forest, st.MainBlue, st.IDAred]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(piv))
    for i, (mth, c) in enumerate(zip(meths, cols)):
        ax.bar(x + (i - 1.5) * 0.2, piv[mth], width=0.2, color=c, label=mth)
    ax.axhline(1, color=st.DarkText, lw=0.8, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels([SERIES_LABEL[k] for k in piv.index])
    ax.set_yscale('log')
    ax.set_ylabel('MASE on the test set (log scale)')
    st.legend_outside_bottom(ax, ncol=4, y=-0.14)
    st.check_no_grey(fig)
    if save:
        st.save_fig('tsa_ch0_benchmarks')
    return tab, piv


def main():
    st.apply()
    V = VALUES
    # --- GDP
    g = fig_gdp()
    V['gdp_first'] = str(g.index[0].year) + ' Q1'
    V['gdp_last_q'] = f'{g.index[-1].year} Q{g.index[-1].quarter}'
    V['gdp_n'] = len(g)
    V['gdp_last_nsa'] = g['nsa'].iloc[-1] / 1000
    V['gdp_last_sca'] = g['sca'].iloc[-1] / 1000
    yr = g['nsa'].groupby(g.index.year).sum()
    full = yr[g['nsa'].groupby(g.index.year).count() == 4]
    V['gdp_year_last'] = int(full.index[-1])
    V['gdp_year_first'] = int(full.index[0])
    V['gdp_mult'] = full.iloc[-1] / full.iloc[0]
    V['gdp_growth_ann'] = 100 * ((full.iloc[-1] / full.iloc[0]) ** (1 / (full.index[-1] - full.index[0])) - 1)
    V['gdp_fall_2009'] = 100 * (full.loc[2009] / full.loc[2008] - 1)
    V['gdp_fall_2020'] = 100 * (full.loc[2020] / full.loc[2019] - 1)
    # --- HICP
    h = fig_hicp()
    inf = h['inflation'].dropna()
    V['hicp_last_date'] = str(h.index[-1].date())
    V['hicp_last'] = h['hicp'].iloc[-1]
    V['infl_last'] = inf.iloc[-1]
    V['infl_first_date'] = str(inf.index[0].date())
    V['infl_first'] = inf.iloc[0]
    V['infl_max'] = inf.max()
    V['infl_max_date'] = str(inf.idxmax().date())
    i2 = inf['2010':]
    V['infl_min10'] = i2.min()
    V['infl_min10_date'] = str(i2.idxmin().date())
    p22 = inf['2021':'2024']
    V['infl_peak22'] = p22.max()
    V['infl_peak22_date'] = str(p22.idxmax().date())
    # --- EUR/RON
    s = fig_eurron()
    V['eurron_first'] = s.iloc[0]
    V['eurron_first_date'] = str(s.index[0].date())
    V['eurron_last'] = s.iloc[-1]
    V['eurron_n'] = len(s)
    V['eurron_min'] = s.min()
    V['eurron_min_date'] = str(s.idxmin().date())
    ds = 100 * np.log(s).diff().dropna()
    V['eurron_maxmove'] = ds.abs().max()
    V['eurron_maxmove_date'] = str(ds.abs().idxmax().date())
    V['eurron_sd'] = ds.std()
    # --- markets
    idx = fig_markets()
    V['bet_mult'] = idx['bet'].iloc[-1] / 100
    V['sp_mult'] = idx['sp500'].iloc[-1] / 100
    V['mk_first'] = str(idx.index[0].date())
    # --- electricity
    e = fig_electricity()
    mm = e.groupby(e.index.month).mean()
    V['elec_first'] = str(e.index[0].date())
    V['elec_last'] = str(e.index[-1].date())
    V['elec_mean'] = e.mean() / 1000
    V['elec_max_month'] = int(mm.idxmax())
    V['elec_min_month'] = int(mm.idxmin())
    V['elec_max_mean'] = mm.max() / 1000
    V['elec_min_mean'] = mm.min() / 1000
    V['elec_ratio'] = mm.max() / mm.min()
    y08 = e[e.index.year == 2008].sum() / 1000
    y25 = e[e.index.year == 2025].sum() / 1000
    V['elec_2008'] = y08
    V['elec_2025'] = y25
    # --- CO2
    c = fig_co2()
    V['co2_first'] = c.iloc[0]
    V['co2_first_date'] = str(c.index[0].date())
    V['co2_last'] = c.iloc[-1]
    V['co2_last_date'] = str(c.index[-1].date())
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    V['co2_slope'] = (c.iloc[-1] - c.iloc[0]) / yrs
    V['co2_n'] = len(c)
    # --- Slutsky
    _, _, up = fig_slutsky()
    V['slutsky_waves'] = up
    # --- decomposition
    y, trend, seas, rem, q = fig_components()
    for k, v in q.items():
        V[f'gdp_S{k}'] = v
        V[f'gdp_S{k}_pct'] = 100 * (v - 1)
    V['gdp_rem_sd'] = 100 * float(np.nanstd(rem.values - 1))
    # 2x4 moving average: a worked example on the first five quarters of a recent year
    yy = (g['nsa'] / 1000)['2024':].iloc[:5]
    V['ma_y'] = [round(float(v), 1) for v in yy.values]
    V['ma_dates'] = [f'{d.year} Q{d.quarter}' for d in yy.index]
    vals = np.round(yy.values, 1)
    V['ma_2x4'] = float((0.5 * vals[0] + vals[1] + vals[2] + vals[3] + 0.5 * vals[4]) / 4)
    # --- STL
    c2, res = fig_stl()
    amp = res.seasonal.groupby(res.seasonal.index.year).agg(lambda x: x.max() - x.min())
    V['co2_amp_first'] = float(amp.iloc[1])
    V['co2_amp_last'] = float(amp.iloc[-1])
    V['co2_seas_max_month'] = int(res.seasonal.groupby(res.seasonal.index.month).mean().idxmax())
    V['co2_seas_min_month'] = int(res.seasonal.groupby(res.seasonal.index.month).mean().idxmin())
    V['co2_resid_sd'] = float(res.resid.std())
    # --- returns
    p, r = fig_returns()
    V['bet_n'] = len(r)
    V['bet_first_date'] = str(r.index[0].date())
    V['bet_sd'] = r.std()
    V['bet_mean'] = r.mean()
    V['bet_min'] = r.min()
    V['bet_min_date'] = str(r.idxmin().date())
    V['bet_max'] = r.max()
    V['bet_max_date'] = str(r.idxmax().date())
    # --- ACF
    out = fig_acf()
    a_c, b_c, n_c = out['CO2 level (monthly)']
    a_e, b_e, n_e = out['Electricity generation (monthly)']
    a_r, b_r, n_r = out['BET daily log returns']
    V['acf_co2_1'], V['acf_co2_36'] = a_c[0], a_c[35]
    V['acf_el_1'], V['acf_el_6'], V['acf_el_12'] = a_e[0], a_e[5], a_e[11]
    V['acf_bet_1'], V['acf_bet_2'] = a_r[0], a_r[1]
    V['band_bet'] = b_r
    V['band_el'] = b_e
    V['acf_bet_out'] = int((np.abs(a_r) > b_r).sum())
    V['acf_absbet_1'] = sample_acf(np.abs(log_returns('bet', '2000-01-01')), 1)[0]
    # --- SES
    ym, alpha_hat = fig_ses()
    V['ses_alpha'] = alpha_hat
    V['ses_first'] = str(ym.index[0].date())
    V['ses_last'] = str(ym.index[-1].date())
    V['ses_last_y'] = ym.iloc[-1]
    # --- forecasts on electricity
    train, test, f, acc = fig_forecast()
    V['fc_test_first'] = str(test.index[0].date())
    V['fc_test_last'] = str(test.index[-1].date())
    V['fc_train_first'] = str(train.index[0].date())
    for k, a in acc.items():
        key = {'Naive': 'nv', 'Seasonal naive': 'snv', 'SES': 'ses', 'Holt-Winters': 'hw'}[k]
        for mtr, v in a.items():
            V[f'fc_{key}_{mtr.lower()}'] = v
    # --- benchmarks
    tab, piv = fig_benchmarks()
    tab.to_csv(os.path.join(TABLE_DIR, 'ch0_accuracy.csv'), index=False)
    for k in piv.index:
        for mth in piv.columns:
            key = {'Naive': 'nv', 'Seasonal naive': 'snv', 'SES': 'ses', 'Holt-Winters': 'hw'}[mth]
            V[f'bm_{k}_{key}'] = float(piv.loc[k, mth])
        V[f'bm_{k}_best'] = str(piv.loc[k].idxmin())
    V['bm_hw_wins'] = int((piv.idxmin(axis=1) == 'Holt-Winters').sum())
    V['bm_snv_beats_hw'] = int((piv['Seasonal naive'] < piv['Holt-Winters']).sum())
    V['bm_hicp_test_first'] = str(hicp_series().index[-TEST_H['hicp']].date())
    for k in V:
        if isinstance(V[k], (np.floating, np.integer)):
            V[k] = V[k].item()
    with open(os.path.join(TABLE_DIR, 'ch0_values.json'), 'w') as fh:
        json.dump(V, fh, indent=1)
    print(json.dumps(V, indent=1))


if __name__ == '__main__':
    main()
