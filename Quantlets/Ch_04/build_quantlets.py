"""
build_quantlets.py -- Quantlet folders of Chapter 4 (TSA): seasonality and forecasting (SARIMA, TBATS, Prophet)
=============================================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/tsa_quantlets.py).
Run:  python3 Quantlets/Ch_04/generate_all_charts.py && python3 Quantlets/Ch_04/seminar4.py
      python3 Quantlets/Ch_04/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 4
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, HERE)
import generate_all_charts as g                 # noqa: E402
from tsa_quantlets import build                 # noqa: E402

SUBMITTED = 'Monday, 5 October 2026'
TITLE = 'Seasonality and forecasting: SARIMA, TBATS, Prophet'
DATA = ('Romanian quarterly real GDP (not adjusted and seasonally adjusted), monthly retail trade, industrial production, '
        'tourism nights and HICP from Eurostat; hourly electricity load of Romania from ENTSO-E (monthly hourly load values); '
        'the airline passengers of Box and Jenkins (R data set via statsmodels)')
NODATA = 'Theoretical models only (no data)'
INSTALL = ("# Optional packages (TBATS, Prophet, pmdarima): installed only if missing (about one minute in Colab)\n"
           "import importlib.util, subprocess, sys\n"
           "for pkg in ['tbats', 'prophet']:\n"
           "    if importlib.util.find_spec(pkg) is None:\n"
           "        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=False)")
CONSTS = ['import logging', 'import matplotlib.dates as mdates',
          'from statsmodels.tsa.stattools import acf, pacf', 'from statsmodels.tsa.statespace.sarimax import SARIMAX',
          'from statsmodels.tsa.arima_process import ArmaProcess', 'from statsmodels.tsa.holtwinters import ExponentialSmoothing',
          'from statsmodels.tsa.seasonal import MSTL', 'from statsmodels.regression.linear_model import OLS',
          'from dateutil.easter import easter, EASTER_ORTHODOX',
          "logging.getLogger('cmdstanpy').setLevel(logging.WARNING)", "logging.getLogger('prophet').setLevel(logging.WARNING)",
          f'SEED = {g.SEED!r}', f'GDP_NSA = {g.GDP_NSA!r}', f'GDP_SCA = {g.GDP_SCA!r}', f'RETAIL = {g.RETAIL!r}',
          f'RETAIL_SCA = {g.RETAIL_SCA!r}', f'FOOD = {g.FOOD!r}', f'IPI = {g.IPI!r}', f'TOUR = {g.TOUR!r}', f'HICP = {g.HICP!r}',
          f'GDP_START = {g.GDP_START!r}', f'ENTSOE = {g.ENTSOE!r}', f'LOAD_FILE = {g.LOAD_FILE!r}', f'LOAD_RAW = {g.LOAD_RAW!r}',
          f'LOAD_YEARS = {g.LOAD_YEARS!r}', f'LOAD_END = {g.LOAD_END!r}',
          f'CV_FIRST, CV_STEP, CV_H = {g.CV_FIRST!r}, {g.CV_STEP!r}, {g.CV_H!r}', 'BAND_COL = st.IDAred',
          "MODEL_COL = {'Seasonal naive': st.Amber, 'ETS': st.Teal, 'SARIMA': st.Purple, 'DHR': st.MainBlue, "
          "'TBATS': st.Forest, 'Prophet': st.Orange, 'Combination': st.IDAred}",
          f'GDP_GRID = {g.GDP_GRID!r}', 'HERE = "."']
CORE = [g.have, g.eurostat_log, g.ro_holidays, g.load_hourly, g.load_daily, g.sample_acf, g.sample_pacf, g.ljung_box,
        g.acf_bars, g.save, g.date_axis, g.polymul, g.sarma_process, g.fit_sarima, g.aicc, g.fourier, g.easter_share,
        g.working_days]
LOAD = [g.load_exog, g.select_orders, g.forecast_models, g.dm_test, g.dm_raw]

QUANTLETS = [
    dict(name='TSA_ch4_seasonal_data',
         desc='Seasonality in Romanian data: real GDP (not adjusted), retail trade, industrial production, tourism nights, '
              'monthly HICP inflation (Eurostat) and daily electricity load (ENTSO-E); a seasonal plot of tourism nights and '
              'the quarterly subseries of GDP; Eurostat seasonally and calendar adjusted (SCA) against unadjusted (NSA) '
              'series of GDP and retail trade, with the implied seasonal factors.',
         keywords='seasonality, seasonal plot, subseries plot, seasonal adjustment, NSA, SCA, Eurostat, GDP, retail trade, tourism, Romania',
         consts=CONSTS, funcs=CORE + [g.airline, g.fig_series, g.fig_seasonal_plot, g.fig_sa_nsa],
         run='print(fig_series())\nprint(fig_seasonal_plot())\nprint(fig_sa_nsa())',
         charts=['tsa_ch4_series', 'tsa_ch4_seasonal_plot', 'tsa_ch4_sa_nsa']),
    dict(name='TSA_ch4_seasonal_roots',
         desc='Seasonal differencing and seasonal unit roots: the roots of 1 - z^4 and 1 - z^12 on the unit circle; the ACF of '
              'Romanian log GDP after regular, seasonal and both differences; the HEGY test (Hylleberg, Engle, Granger and Yoo, '
              '1990) with critical values by simulation, and the OCSB and Canova-Hansen decisions of pmdarima for five series.',
         keywords='seasonal difference, seasonal unit root, HEGY, Canova-Hansen, OCSB, unit circle, GDP, Romania',
         consts=CONSTS, funcs=CORE + [g.fig_unit_circle, g.fig_gdp_differencing, g.hegy_stats, g.hegy_critical, g.seasonal_tests],
         run='print(fig_unit_circle())\nprint(fig_gdp_differencing())\nprint(seasonal_tests(save_csv=False))',
         charts=['tsa_ch4_unit_circle', 'tsa_ch4_gdp_diff'], install=INSTALL),
    dict(name='TSA_ch4_sarima_theory',
         desc='Theoretical ACF and PACF of three monthly seasonal models: a seasonal AR(1) with Phi = 0.8, a seasonal MA(1) with '
              'Theta = -0.6 and the airline model with theta = -0.4 and Theta = -0.6 (the multiplicative polynomials are expanded '
              'and passed to ArmaProcess).',
         keywords='SARIMA, seasonal AR, seasonal MA, airline model, ACF, PACF, multiplicative model',
         consts=CONSTS, funcs=CORE + [g.fig_theory_acf], run='print(fig_theory_acf())', charts=['tsa_ch4_theory_acf'], data=NODATA),
    dict(name='TSA_ch4_airline',
         desc='The airline model of Box and Jenkins: the monthly international airline passengers 1949-1960, the ACF and PACF of '
              'the doubly differenced log series, SARIMA(0,1,1)(0,1,1)_12 estimated on 1949-1958 and 24 forecasts out of sample '
              'with 95% intervals, compared with the seasonal naive forecast.',
         keywords='airline model, SARIMA, Box-Jenkins, airline passengers, forecast, seasonal naive, MAPE',
         consts=CONSTS, funcs=CORE + [g.airline, g.fig_airline, g.fig_airline_forecast],
         run='print(fig_airline())\nprint(fig_airline_forecast())', charts=['tsa_ch4_airline', 'tsa_ch4_airline_fc']),
    dict(name='TSA_ch4_gdp_case',
         desc='SARIMA for Romanian real GDP, not adjusted (Eurostat, 2000 onwards): ten SARIMA(p,1,q)(P,1,Q)_4 models with AICc, '
              'BIC and Ljung-Box, residual diagnostics and forecasts 8 quarters ahead of the BIC model, and a rolling-origin '
              'comparison of the airline model, ETS and the seasonal naive method with Diebold-Mariano tests.',
         keywords='SARIMA, GDP, Romania, Eurostat, AICc, BIC, Ljung-Box, forecast, rolling origin, Diebold-Mariano',
         consts=CONSTS, funcs=CORE + LOAD + [g.gdp_models, g.parse_model, g.fig_gdp_diag, g.fig_gdp_forecast, g.gdp_cv],
         run='T = gdp_models(save_csv=False)\nprint(T["best_aicc"], T["best_bic"])\no, so = parse_model(T["best_bic"])\n'
             'print(fig_gdp_diag(o, so))\nprint(fig_gdp_forecast(o, so))\nprint(gdp_cv()["tab"])',
         charts=['tsa_ch4_gdp_diag', 'tsa_ch4_gdp_fc'], extra=['ch4_gdp_models.csv']),
    dict(name='TSA_ch4_calendar_fourier',
         desc='Fourier terms and calendar effects: Fourier approximations (K = 1, 2, 3, 6) of the monthly seasonal pattern of '
              'tourism nights; Orthodox Easter dates 2000-2030; regression with SARIMA(0,1,1)(0,1,1)_12 errors of Romanian food '
              'and total retail trade on the Easter regressor (share of the 10 days before Easter in each month), working days '
              'and lockdown outliers.',
         keywords='Fourier terms, harmonics, calendar effects, Orthodox Easter, working days, regARIMA, retail trade, Romania',
         consts=CONSTS, funcs=CORE + [g.fig_fourier, g.easter_regression, g.fig_easter],
         run='print(fig_fourier())\nprint(fig_easter())', charts=['tsa_ch4_fourier', 'tsa_ch4_easter']),
    dict(name='TSA_ch4_multiple_seasonality',
         desc='Multiple seasonality in the hourly electricity load of Romania (ENTSO-E, 2022-2026): three weeks of hourly data '
              'and the ACF up to 400 hours; MSTL with periods 24 and 168; daily load with public holidays and the mean by day '
              'of the week; a dynamic harmonic regression (annual Fourier terms, weekday and holiday dummies, ARIMA errors).',
         keywords='multiple seasonality, electricity load, ENTSO-E, MSTL, dynamic harmonic regression, holidays, Romania',
         consts=CONSTS, funcs=CORE + LOAD + [g.fig_load_hourly, g.fig_mstl, g.fig_load_daily, g.dhr_fit],
         run='print(fig_load_hourly())\nprint(fig_mstl())\nprint(fig_load_daily())\nprint(dhr_fit())',
         charts=['tsa_ch4_load_hourly', 'tsa_ch4_mstl', 'tsa_ch4_load_daily']),
    dict(name='TSA_ch4_tbats_prophet',
         desc='TBATS (De Livera, Hyndman and Snyder, 2011) and Prophet (Taylor and Letham, 2018) for the daily electricity load '
              'of Romania: the TBATS configuration chosen by AIC (periods 7 and 365.25) and the Prophet components (trend with '
              'changepoints, weekly and yearly seasonality, Romanian public holidays).',
         keywords='TBATS, Prophet, trigonometric seasonality, changepoints, holidays, electricity load, Romania',
         consts=CONSTS, funcs=CORE + [g.fig_prophet_components, g.tbats_fit],
         run='print(fig_prophet_components())\nprint(tbats_fit())', charts=['tsa_ch4_prophet'], install=INSTALL),
    dict(name='TSA_ch4_evaluation',
         desc='Forecast evaluation for the daily electricity load of Romania: time-series cross-validation with an expanding '
              'window (forecast origins every 14 days over the last year, horizon 14 days) of the seasonal naive method, ETS, '
              'SARIMA, a dynamic harmonic regression, TBATS, Prophet and their average; MAE, RMSE, MASE, Diebold-Mariano tests '
              'with the Harvey-Leybourne-Newbold correction, and the combination of two forecasts as a function of the weight. '
              'FAST = True uses an origin every 56 days.',
         keywords='time-series cross-validation, rolling origin, MASE, Diebold-Mariano, forecast combination, TBATS, Prophet, DHR',
         consts=CONSTS + ['FAST = True   # True: an origin every 56 days (a few minutes); False: every 14 days, as in the slides'],
         funcs=CORE + LOAD + [g.load_cv, g.cv_summary, g.fig_cv_scheme, g.fig_cv_results, g.fig_load_forecasts, g.fig_combination],
         run='print(fig_cv_scheme())\nE, orders = load_cv(step=56 if FAST else CV_STEP, save_csv=False)\n'
             'tab, dm, best, scale, byh = cv_summary(E)\nprint(tab.round(4))\nfig_cv_results(tab, byh)\n'
             'print(fig_load_forecasts(E))\nprint(fig_combination(E, "DHR", "Prophet") if "Prophet" in tab.index else "")',
         charts=['tsa_ch4_cv_scheme', 'tsa_ch4_cv_results', 'tsa_ch4_load_fc', 'tsa_ch4_combination'], install=INSTALL),
]

try:                                   # seminar code
    import seminar4 as s
    QUANTLETS.append(dict(
        name='TSA_ch4_seminar',
        desc='Seminar 4 of Time Series Analysis: seasonal differences and seasonal naive forecasts, SARIMA polynomials and the '
             'ACF of the airline model, moment estimates, MASE and the Diebold-Mariano test, forecast combination; the airline '
             'model for Romanian GDP, industrial production with working days, cross-validation of daily electricity load, '
             'seasonality in monthly HICP inflation, tourism nights after the pandemic.',
        keywords='seminar, seasonal difference, SARIMA, airline model, working days, cross-validation, Diebold-Mariano, '
                 'forecast combination, GDP, electricity load, HICP, Romania',
        consts=CONSTS, funcs=CORE + LOAD + [s.seasonal_differences, s.airline_acf, s.dm_by_hand, s.b1_gdp, s.b3_load_cv],
        run="print(seasonal_differences([10, 14, 16, 18, 12, 16, 19, 21, 14]))\nprint(airline_acf(-0.5, -0.6, s=4))\n"
            "print(dm_by_hand([0.4, -0.6, 0.9, -0.3, 0.5, -0.8], [1.1, -0.9, 0.7, -1.2, 0.8, -1.0], scale=1.0))\n"
            "print(b1_gdp())\nprint(b3_load_cv())",
        charts=['ch4_sem_b1', 'ch4_sem_b3']))
except ImportError:
    pass

if __name__ == '__main__':
    only = sys.argv[1:]
    for q in QUANTLETS:
        if not only or q['name'] in only:
            build(q, 4, TITLE, HERE, data=q.get('data', DATA), submitted=SUBMITTED, install=q.get('install', ''))
