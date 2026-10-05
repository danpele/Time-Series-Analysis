"""
build_quantlets.py -- Quantlet folders of Chapter 3 (TSA): unit roots and ARIMA models
=====================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/tsa_quantlets.py).
Run:  python3 Quantlets/Ch_03/generate_all_charts.py && python3 Quantlets/Ch_03/seminar3.py
      python3 Quantlets/Ch_03/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 3
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, HERE)
import generate_all_charts as g                 # noqa: E402
from tsa_quantlets import build_all             # noqa: E402

SUBMITTED = 'Monday, 5 October 2026'
DATA = ('Daily market data from EODHD (BET and S&P 500 since 2000), data/market of the TSA repository; EUR/RON reference '
        'rate of the BNR; Romanian quarterly real GDP and monthly HICP from Eurostat; US real GDP (GDPC1) from FRED; '
        'the Nile data set of statsmodels')
NODATA = 'Simulated data only (no market data)'
CONSTS = ['import itertools', 'import statsmodels.api as sm',
          'from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss, zivot_andrews',
          'from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit',
          'from statsmodels.tsa.arima.model import ARIMA', 'from statsmodels.tsa.arima_process import ArmaProcess',
          'from statsmodels.stats.diagnostic import acorr_ljungbox',
          f'SEED = {g.SEED!r}', f'GDP_SA = {g.GDP_SA!r}', f'HICP = {g.HICP!r}', f'GDP_START = {g.GDP_START!r}',
          f'START = {g.START!r}', 'BAND_COL = st.IDAred', f'CASES = {g.CASES!r}']
CORE = [g.ro_gdp, g.ro_hicp, g.ro_inflation, g.us_gdp, g.quarterly, g.monthly, g.sample_acf, g.sample_pacf, g.acf_bars,
        g.save, g.years_axis, g.adf_test, g.pp_test, g.kpss_test, g.za_test, g.verdict, g.df_tau_batch, g.random_walks,
        g.ljung_box]
ARIMA_F = [g.arima_grid, g.best_order, g.gdp_series, g.gdp_order, g.gdp_fit]

QUANTLETS = [
    dict(name='TSA_ch3_trends',
         desc='Deterministic and stochastic trends: four trending series (Romanian real GDP and HICP from Eurostat, the '
              'EUR/RON reference rate of the BNR, BET and S&P 500 log prices); a trend-stationary and a difference-stationary '
              'series driven by the same shocks, and the persistence of one shock; a random walk detrended by a straight '
              'line (spurious cycles, Nelson and Kang 1981) with a Monte Carlo of the trend t-statistic.',
         keywords='deterministic trend, stochastic trend, trend-stationary, difference-stationary, random walk with drift, '
                  'detrending, spurious cycles, GDP, HICP, EUR/RON, BET, S&P 500',
         consts=CONSTS, funcs=CORE + [g.fig_four_series, g.fig_ts_ds, g.fig_detrend],
         run='print(fig_four_series())\nprint(fig_ts_ds())\nprint(fig_detrend())',
         charts=['tsa_ch3_four_series', 'tsa_ch3_ts_ds', 'tsa_ch3_detrend']),
    dict(name='TSA_ch3_spurious_regression',
         desc='Spurious regression (Granger and Newbold 1974): regression of one random walk on another, independent one, '
              'for T = 25 to 1000; rejection rate of the t-test, R^2 and Durbin-Watson in levels and in differences; '
              'Romanian log HICP regressed on the log S&P 500, in levels and in monthly changes.',
         keywords='spurious regression, nonsense correlation, Granger-Newbold, Durbin-Watson, R squared, random walk, '
                  'Monte Carlo, HICP, S&P 500',
         consts=CONSTS, funcs=CORE + [g.spurious_mc, g.fig_spurious_mc, g.ols_summary, g.fig_spurious_real],
         run='print(fig_spurious_mc())\nprint(fig_spurious_real())',
         charts=['tsa_ch3_spurious_mc', 'tsa_ch3_spurious_real']),
    dict(name='TSA_ch3_dickey_fuller',
         desc='The Dickey-Fuller test: the null distribution of the t-statistic in the three specifications (no constant, '
              'constant, constant and trend) by simulation, against N(0, 1) and the MacKinnon critical values; size and power '
              'for T = 100, 250, 500; a worked example on Romanian real GDP.',
         keywords='Dickey-Fuller, augmented Dickey-Fuller, ADF, unit root, critical values, MacKinnon, size, power, '
                  'Monte Carlo, GDP, Romania',
         consts=CONSTS, funcs=CORE + [g.fig_df_dist, g.fig_adf_power, g.df_by_hand],
         run='print(fig_df_dist())\nprint(fig_adf_power())\nprint(df_by_hand())',
         charts=['tsa_ch3_df_dist', 'tsa_ch3_adf_power']),
    dict(name='TSA_ch3_unit_root_tests',
         desc='ADF, Phillips-Perron (written out with the Newey-West long-run variance) and KPSS tests on fourteen real '
              'series: Romanian GDP, HICP and inflation, US real GDP, EUR/RON, BET and S&P 500 log prices and returns, the '
              'Nile flow; the joint ADF-KPSS verdict; KPSS partial sums of a stationary AR(1) and of a random walk.',
         keywords='unit root test, ADF, Phillips-Perron, KPSS, long-run variance, Newey-West, stationarity test, '
                  'integration order, Romania, EUR/RON, BET, S&P 500',
         consts=CONSTS, funcs=CORE + [g.fig_kpss_sums, g.real_series, g.unit_root_table],
         run='print(fig_kpss_sums())\nU = unit_root_table()\nprint(pd.DataFrame({k: {"ADF": v["adf"]["stat"], "ADF p": v["adf"]["p"], '
             '"PP": v["pp"]["stat"], "KPSS": v["kpss"]["stat"], "verdict": v["verdict"]} for k, v in U.items()}).T)',
         charts=['tsa_ch3_kpss_sums'], extra=['ch3_unit_root_tests.csv']),
    dict(name='TSA_ch3_structural_breaks',
         desc='Structural breaks and unit roots (Perron 1989; Zivot and Andrews 1992): a stationary AR(1) around a mean that '
              'shifts once, the Dickey-Fuller rejection rate with and without the break, the Zivot-Andrews search over break '
              'dates; Zivot-Andrews on the Nile flow (break in 1898) and on the month-end EUR/RON rate.',
         keywords='structural break, Perron, Zivot-Andrews, level shift, unit root, Nile, EUR/RON',
         consts=CONSTS, funcs=CORE + [g.fig_breaks_sim, g.fig_breaks_real],
         run='print(fig_breaks_sim())\nprint(fig_breaks_real())',
         charts=['tsa_ch3_breaks_sim', 'tsa_ch3_breaks_real']),
    dict(name='TSA_ch3_arima_gdp',
         desc='An ARIMA model for Romanian quarterly real GDP (Eurostat, seasonally adjusted, since 2000): identification '
              '(ACF, PACF of the growth rate), ARIMA(p,1,q) with drift by AICc on two samples, residual diagnostics, '
              'forecasts for 12 quarters against a trend-stationary model, over-differencing.',
         keywords='ARIMA, Box-Jenkins, AICc, drift, residual diagnostics, Ljung-Box, forecast intervals, over-differencing, '
                  'GDP, Romania',
         consts=CONSTS, funcs=CORE + ARIMA_F + [g.fig_gdp_ident, g.gdp_models, g.fig_gdp_diag, g.fig_gdp_forecast,
                                                g.fig_overdiff_gdp],
         run='print(fig_gdp_ident())\nprint(gdp_models())\nprint(fig_gdp_diag())\nprint(fig_gdp_forecast())\nprint(fig_overdiff_gdp())',
         charts=['tsa_ch3_gdp_ident', 'tsa_ch3_gdp_diag', 'tsa_ch3_gdp_forecast', 'tsa_ch3_overdiff_gdp']),
    dict(name='TSA_ch3_forecast_intervals',
         desc='Forecast intervals of ARIMA models from the psi weights: half-width of the 95% interval against the horizon '
              'for AR(1), the random walk, ARIMA(1,1,0), ARIMA(0,1,1) and ARIMA(0,2,0); a worked ARIMA(1,1,0) example.',
         keywords='forecast interval, psi weights, ARIMA, random walk, horizon, forecast error variance',
         consts=CONSTS, funcs=CORE + [g.psi_weights, g.fig_interval_width, g.arima110_by_hand],
         run='print(fig_interval_width())\nprint(arima110_by_hand())', charts=['tsa_ch3_interval_width'], data=NODATA),
    dict(name='TSA_ch3_arima_in_practice',
         desc='ARIMA in practice: Romanian 12-month HICP inflation with d = 0 and d = 1 (AICc within each d, KPSS choice of d '
              'as in Hyndman and Khandakar 2008) and their 24-month forecasts; EUR/RON one-day and 20-day forecasts of an '
              'ARIMA(1,1,0) against the random walk (Meese and Rogoff 1983).',
         keywords='automatic ARIMA, Hyndman-Khandakar, KPSS, AICc, inflation, Romania, EUR/RON, random walk benchmark, '
                  'Meese-Rogoff, RMSE',
         consts=CONSTS, funcs=CORE + ARIMA_F + [g.fig_inflation_d, g.fig_eurron_forecast],
         run='print(fig_inflation_d())\nprint(fig_eurron_forecast())',
         charts=['tsa_ch3_inflation_d', 'tsa_ch3_eurron_forecast']),
    dict(name='TSA_ch3_us_gdp',
         desc='A classic series: US real GDP (FRED GDPC1, since 1947). A trend-stationary model (linear trend with AR(2) '
              'errors) and an ARIMA(1,1,0) with drift, both estimated up to 2007Q4, forecast to the end of the data '
              'against the outcome.',
         keywords='US GDP, trend-stationary, difference-stationary, Great Recession, permanent shock, forecast, ARIMA',
         consts=CONSTS, funcs=CORE + [g.fig_us_gdp], run='print(fig_us_gdp())', charts=['tsa_ch3_us_gdp'],
         data='FRED series GDPC1 (US real GDP), read online'),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar3 as s
    QUANTLETS.append(dict(
        name='TSA_ch3_seminar',
        desc='Seminar 3 of Time Series Analysis: a Dickey-Fuller statistic by hand, ADF specifications, ARIMA(1,1,0) and '
             'ARIMA(0,1,1) forecasts with intervals; ADF, PP and KPSS on the BET, the S&P 500 and EUR/RON; an ARIMA for '
             'Romanian real GDP with 8-quarter forecasts; Romanian inflation with d = 0 and d = 1; Zivot-Andrews breaks.',
        keywords='seminar, unit root, Dickey-Fuller, KPSS, ARIMA, forecast, BET, GDP, inflation, Romania, Zivot-Andrews',
        consts=CONSTS, funcs=CORE + ARIMA_F + [g.psi_weights, s.df_by_hand, s.adf_choice, s.arima_by_hand, s.ar_roots,
                                               s.unit_root_set, s.b1_tests, s.b3_gdp],
        run="print(df_by_hand(-0.061, 0.025, 120, 'c'))\nprint(arima_by_hand(ar=(0.6,), d=1, y=(103.0, 108.0), sigma=2.0))\n"
            "print(ar_roots([1.5, -0.5]))\nprint(b1_tests('bet', fname='ch3_sem_b1'))\nprint(b3_gdp())",
        charts=['ch3_sem_b1', 'ch3_sem_b3']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 3, 'Unit roots and ARIMA models', HERE, data=DATA, submitted=SUBMITTED)
